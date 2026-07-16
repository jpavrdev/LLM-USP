"""Baixa datasets Q&A em PT-BR e formata pra mesclar com o corpus de treino existente.

Uso (a partir da raiz do projeto):
    webapp/.venv/bin/python tucano/fetch_datasets.py

Formato de saída (compatível com o loop de leitura do notebook gpt-v6/v7.ipynb):
    Pergunta: <instrução>
    <contexto opcional>

    Resposta: <resposta>
    <|endoftext|>
    Pergunta: ...

Os arquivos ficam em ../data/qa_*.txt e passam a ser lidos na próxima rodada de treino.
"""

from pathlib import Path
import random

from datasets import load_dataset


OUT_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_DIR.mkdir(exist_ok=True)


def format_example(instruction: str, context: str, response: str) -> str:
    """Formata um trio instrução/contexto/resposta como texto PT-BR."""
    parts = [f"Pergunta: {instruction.strip()}"]
    if context and context.strip():
        parts.append(context.strip())
    parts.append("")
    parts.append(f"Resposta: {response.strip()}")
    return "\n".join(parts) + "\n<|endoftext|>\n"


def process_dolly(out_path: Path, limit: int | None = None):
    """Dolly 15k PT — escrito por humanos na Databricks, traduzido via LibreTranslate."""
    print(f"[dolly] baixando Gustrd/dolly-15k-libretranslate-pt ...")
    ds = load_dataset("Gustrd/dolly-15k-libretranslate-pt", split="train")
    print(f"[dolly] {len(ds)} exemplos totais")

    # Campos: instruction, context, response, category
    rows = list(ds)
    if limit:
        random.seed(42)
        random.shuffle(rows)
        rows = rows[:limit]

    kept = 0
    skipped = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for row in rows:
            instr = (row.get("instruction") or "").strip()
            ctx = (row.get("context") or "").strip()
            resp = (row.get("response") or "").strip()
            # pula exemplos vazios ou quase vazios
            if len(instr) < 3 or len(resp) < 3:
                skipped += 1
                continue
            f.write(format_example(instr, ctx, resp))
            kept += 1

    size_mb = out_path.stat().st_size / 1e6
    print(f"[dolly] gravado em {out_path.name}: {kept} exemplos, {size_mb:.2f} MB (pulados: {skipped})")


def process_alpaca(out_path: Path, limit: int = 15000):
    """Alpaca PT-BR — 52k instruções do Stanford traduzidas via OPUS-MT. Samplear só `limit`."""
    print(f"[alpaca] baixando dominguesm/alpaca-data-pt-br ...")
    ds = load_dataset("dominguesm/alpaca-data-pt-br", split="train")
    print(f"[alpaca] {len(ds)} exemplos totais, samplear {limit}")

    rows = list(ds)
    random.seed(42)
    random.shuffle(rows)
    rows = rows[:limit]

    kept = 0
    skipped = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for row in rows:
            instr = (row.get("instruction") or "").strip()
            # Alpaca tem campo `input` em vez de `context`, e `output` em vez de `response`
            ctx = (row.get("input") or "").strip()
            resp = (row.get("output") or "").strip()
            if len(instr) < 3 or len(resp) < 3:
                skipped += 1
                continue
            f.write(format_example(instr, ctx, resp))
            kept += 1

    size_mb = out_path.stat().st_size / 1e6
    print(f"[alpaca] gravado em {out_path.name}: {kept} exemplos, {size_mb:.2f} MB (pulados: {skipped})")


if __name__ == "__main__":
    process_dolly(OUT_DIR / "qa_dolly_ptbr.txt")
    process_alpaca(OUT_DIR / "qa_alpaca_ptbr.txt", limit=15000)
    print("\n=== Pronto ===")
    print(f"Arquivos escritos em: {OUT_DIR}")
    print("Próximo passo: rodar gpt-v7.ipynb (ou criar gpt-v8.ipynb) do zero pra treinar com o novo corpus.")
