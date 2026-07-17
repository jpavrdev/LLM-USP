"""Constrói o índice RAG a partir do Wikipedia-PT.

Pipeline:
  1. Baixa subset do dataset wikimedia/wikipedia (config 20231101.pt)
  2. Filtra artigos muito curtos (stubs)
  3. Chunkar artigos por parágrafos, agrupando até ~300 tokens
  4. Embeda chunks via intfloat/multilingual-e5-base
  5. Cria índice FAISS (inner product com vetores normalizados = cos sim)
  6. Salva índice + metadata em ../rag/

Requer: sentence-transformers, faiss-cpu, datasets (já instalados)
GPU recomendada (CUDA) pra acelerar embeddings em ~10x.

Uso (a partir da raiz do projeto):
    webapp/.venv/bin/python rag/build_rag_index.py [--max-articles N]
"""
import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
from datasets import load_dataset


ROOT = Path(__file__).resolve().parent
INDEX_DIR = ROOT.parent / "rag"
INDEX_DIR.mkdir(exist_ok=True)

DATASET = "wikimedia/wikipedia"
CONFIG = "20231101.pt"
EMBED_MODEL = "intfloat/multilingual-e5-base"
EMBED_DIM = 768
MIN_ARTICLE_CHARS = 800       # filtra stubs
MAX_CHUNK_TOKENS_APPROX = 300 # ~ tokens (heurística por chars)
CHARS_PER_TOKEN = 4           # média PT-BR
MAX_CHUNK_CHARS = MAX_CHUNK_TOKENS_APPROX * CHARS_PER_TOKEN


def chunk_article(title: str, text: str) -> list[dict]:
    """Divide um artigo em chunks de ~300 tokens, agrupando parágrafos."""
    # Limpa marcações remanescentes simples
    text = re.sub(r"\n{3,}", "\n\n", text)
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip() and len(p.strip()) > 30]
    chunks = []
    buf: list[str] = []
    buf_len = 0
    for p in paragraphs:
        if buf_len + len(p) > MAX_CHUNK_CHARS and buf:
            chunks.append("\n".join(buf))
            buf, buf_len = [], 0
        buf.append(p)
        buf_len += len(p)
    if buf:
        chunks.append("\n".join(buf))
    # Cabeçalho do artigo no início do primeiro chunk pra ajudar retrieval
    return [{"title": title, "chunk_idx": i, "text": ck} for i, ck in enumerate(chunks)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-articles", type=int, default=100_000,
                    help="Limite de artigos a processar (default 100k)")
    ap.add_argument("--max-chunks-per-article", type=int, default=3,
                    help="Máximo de chunks por artigo (os primeiros têm o cabeçalho/intro, mais relevantes)")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    print(f"[rag] carregando dataset {DATASET}/{CONFIG}...")
    ds = load_dataset(DATASET, CONFIG, split="train")
    print(f"[rag] {len(ds):,} artigos no total")

    # Filtra: ordena por tamanho, pega os maiores (mais ricos)
    print(f"[rag] filtrando artigos com >= {MIN_ARTICLE_CHARS} chars...")
    valid_indices = []
    for i, ex in enumerate(ds):
        if len(ex["text"]) >= MIN_ARTICLE_CHARS:
            valid_indices.append(i)
    print(f"[rag] {len(valid_indices):,} artigos passaram no filtro de tamanho")

    if args.max_articles and args.max_articles < len(valid_indices):
        # Pega os MAIORES (proxy de "mais informativos")
        print(f"[rag] selecionando os {args.max_articles:,} artigos mais longos...")
        valid_indices = sorted(valid_indices, key=lambda i: -len(ds[i]["text"]))[:args.max_articles]
        print(f"[rag] amostra final: {len(valid_indices):,}")

    # Chunking
    print("\n[rag] chunkando artigos...")
    t0 = time.time()
    all_chunks: list[dict] = []
    for ci, idx in enumerate(valid_indices):
        ex = ds[idx]
        for c in chunk_article(ex["title"], ex["text"])[:args.max_chunks_per_article]:
            c["url"] = ex.get("url", "")
            c["doc_idx"] = idx
            all_chunks.append(c)
        if (ci + 1) % 5000 == 0:
            elapsed = time.time() - t0
            print(f"  {ci+1:,}/{len(valid_indices):,} ({len(all_chunks):,} chunks, {elapsed:.1f}s)", flush=True)
    print(f"[rag] total: {len(all_chunks):,} chunks em {time.time()-t0:.0f}s")

    # Carrega modelo de embeddings
    print(f"\n[rag] carregando modelo de embeddings: {EMBED_MODEL}...")
    import torch
    from sentence_transformers import SentenceTransformer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embed_model = SentenceTransformer(EMBED_MODEL, device=device)
    print(f"[rag] modelo no device: {device}")

    # E5 espera prefix "passage: " nos textos indexados, "query: " nas queries
    texts_to_embed = [f"passage: {c['title']}: {c['text']}" for c in all_chunks]

    print(f"\n[rag] gerando embeddings ({len(texts_to_embed):,} chunks, batch={args.batch_size})...")
    t0 = time.time()
    embeddings = embed_model.encode(
        texts_to_embed,
        batch_size=args.batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    print(f"[rag] embeddings: shape {embeddings.shape}, dtype {embeddings.dtype}, em {time.time()-t0:.0f}s")

    # Salva índice FAISS
    print("\n[rag] criando índice FAISS (Inner Product = cos similarity)...")
    import faiss
    embeddings = embeddings.astype(np.float32)
    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(embeddings)
    faiss_path = INDEX_DIR / "wiki_pt.faiss"
    faiss.write_index(index, str(faiss_path))
    print(f"[rag] FAISS salvo: {faiss_path} ({faiss_path.stat().st_size/1e6:.1f} MB)")

    # Salva metadata (texto + título de cada chunk) em parquet pra busca rápida
    import pyarrow as pa, pyarrow.parquet as pq
    table = pa.table({
        "title": [c["title"] for c in all_chunks],
        "chunk_idx": [c["chunk_idx"] for c in all_chunks],
        "text": [c["text"] for c in all_chunks],
        "url": [c["url"] for c in all_chunks],
    })
    meta_path = INDEX_DIR / "wiki_pt_meta.parquet"
    pq.write_table(table, str(meta_path))
    print(f"[rag] metadata salvo: {meta_path} ({meta_path.stat().st_size/1e6:.1f} MB)")

    # Manifest do índice
    manifest = {
        "dataset": f"{DATASET}/{CONFIG}",
        "embed_model": EMBED_MODEL,
        "embed_dim": EMBED_DIM,
        "n_chunks": len(all_chunks),
        "n_articles": len(valid_indices),
        "min_article_chars": MIN_ARTICLE_CHARS,
        "max_chunk_chars": MAX_CHUNK_CHARS,
    }
    manifest_path = INDEX_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"[rag] manifest: {manifest_path}")
    print(f"\n[rag] DONE — índice pronto em {INDEX_DIR}")


if __name__ == "__main__":
    main()
