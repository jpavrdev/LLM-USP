"""Baixa e inspeciona o dataset TucanoBR/Tucano-SFT.

Objetivo: confirmar a estrutura antes de adaptar o script de treino.
"""
from datasets import load_dataset
import json

print("[inspect] baixando/carregando TucanoBR/Tucano-SFT (primeiro split disponível)...")
ds = load_dataset("TucanoBR/Tucano-SFT", split="train", streaming=False)

print(f"\n[inspect] total de exemplos: {len(ds):,}")
print(f"[inspect] features: {ds.features}")
print()

# Amostras
print("=" * 70)
print("3 AMOSTRAS DO DATASET")
print("=" * 70)
for i in range(3):
    ex = ds[i]
    print(f"\n--- Exemplo {i} ---")
    for k, v in ex.items():
        if isinstance(v, list):
            print(f"  {k}:")
            for item in v[:4]:
                if isinstance(item, dict):
                    content_preview = str(item.get('content', ''))[:200].replace('\n', ' ')
                    role = item.get('role', '?')
                    print(f"    - {role}: {content_preview}")
                else:
                    print(f"    - {str(item)[:200]}")
        else:
            print(f"  {k}: {str(v)[:200]}")

# Distribuição de tamanhos
import numpy as np
print("\n" + "=" * 70)
print("ESTATÍSTICAS DE TAMANHO")
print("=" * 70)
# Sample first 1000 pra não demorar
n_msgs = []
total_chars = []
for i in range(min(1000, len(ds))):
    conv = ds[i].get('conversations', [])
    n_msgs.append(len(conv))
    total_chars.append(sum(len(m.get('content', '')) for m in conv))
print(f"Nº de mensagens por conversa (amostra 1000): min={min(n_msgs)}, max={max(n_msgs)}, mediana={int(np.median(n_msgs))}")
print(f"Chars totais por conversa: min={min(total_chars)}, max={max(total_chars)}, mediana={int(np.median(total_chars))}")

# Origens (source metadata)
print("\n" + "=" * 70)
print("DISTRIBUIÇÃO POR FONTE (amostra 5000)")
print("=" * 70)
sources = {}
for i in range(min(5000, len(ds))):
    ex = ds[i]
    # metadata pode estar em diferentes locais
    src = ex.get('source') or ex.get('metadata', {}).get('source', 'unknown')
    sources[src] = sources.get(src, 0) + 1
for s, c in sorted(sources.items(), key=lambda x: -x[1]):
    print(f"  {s}: {c}")
