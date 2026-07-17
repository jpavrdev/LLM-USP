"""Recuperação RAG sobre o índice da Wikipedia-PT (gerado por build_rag_index.py).

Carrega uma vez o índice FAISS + metadados + modelo e5 (na CPU) e expõe
retrieve(query) -> trechos relevantes pra fundamentar a resposta do modelo.
"""
from pathlib import Path

import faiss
import numpy as np

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
INDEX_DIR = PROJECT_ROOT / "rag"

# Devem bater com build_rag_index.py.
EMBED_MODEL = "intfloat/multilingual-e5-base"
EMBED_DIM = 768

MIN_SCORE = 0.45
DEFAULT_K = 2
MAX_CONTEXT_CHARS = 1600


class Retriever:
    def __init__(self, index_dir=INDEX_DIR, k=DEFAULT_K, min_score=MIN_SCORE):
        import pyarrow.parquet as pq
        from sentence_transformers import SentenceTransformer

        self.index_dir = Path(index_dir)
        self.k = k
        self.min_score = min_score

        self.index = faiss.read_index(str(self.index_dir / "wiki_pt.faiss"))
        meta = pq.read_table(str(self.index_dir / "wiki_pt_meta.parquet")).to_pylist()
        self.titles = [m["title"] for m in meta]
        self.texts = [m["text"] for m in meta]
        self.urls = [m.get("url", "") for m in meta]

        # e5 na CPU: em runtime só codifica a query (~dezenas de ms), sem contender
        # a VRAM com o Tucano-1b1 que fica na GPU.
        self.embed_model = SentenceTransformer(EMBED_MODEL, device="cpu")
        print(f"[rag] retriever pronto: {self.index.ntotal:,} chunks, e5 na CPU, min_score={min_score}")

    def retrieve(self, query, k=None, min_score=None):
        k = self.k if k is None else k
        min_score = self.min_score if min_score is None else min_score
        qv = self.embed_model.encode(
            [f"query: {query}"],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)
        scores, idxs = self.index.search(qv, k)

        out = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx < 0 or score < min_score:
                continue
            out.append({
                "title": self.titles[idx],
                "text": self.texts[idx],
                "url": self.urls[idx],
                "score": float(score),
            })
        return out
