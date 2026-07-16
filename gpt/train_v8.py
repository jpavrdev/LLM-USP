"""Treino do GPT PT-BR v8 — livros + datasets Q&A (Dolly + Alpaca).

Script standalone equivalente ao gpt-v8.ipynb, pra rodar em background
sem depender do Jupyter. Continua do checkpoint_gpt_v8.pth se existir,
senão treina do zero. Pra começar do V7: cp checkpoint_gpt_v7.pth checkpoint_gpt_v8.pth

Uso (a partir da raiz do projeto):
    ./webapp/.venv/bin/python gpt/train_v8.py
"""

import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # sem display
import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F
import tiktoken

# Arquitetura V8 + constantes compartilhadas (definidas sem side-effects).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import (
    GPTLanguageModel,
    device,
    block_size,
    n_embd,
    n_head,
    n_layer,
    dropout,
    vocab_size,
)

# ------------------------------------------------------------
# Hiperparâmetros de treino (mesmo setup do V6/V7 pra compatibilidade)
# ------------------------------------------------------------
batch_size = 16
# Warm-start de V7 + corpus Q&A: fine-tuning, não treino do zero.
# Hiperparâmetros ajustados após observarmos overfit em step 200 com LR=3e-4.
max_iters = 3000          # overfit aparece cedo; preferimos parar antes
learning_rate = 5e-5      # LR típico de fine-tuning (1/6 do valor anterior)
eval_iters = 200
eval_batches = 50
early_stop_patience = 3   # para se val não melhorar por N evals seguidos

# Caminhos resolve­dos a partir da raiz do projeto (script roda de qualquer CWD).
ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_FILE = ROOT / "checkpoint_gpt_v8.pth"
BEST_CHECKPOINT = ROOT / "checkpoint_gpt_v8_best.pth"   # melhor loss_val visto
FINAL_MODEL = ROOT / "gpt_ptbr_v8.pth"
PLOT_FILE = ROOT / "outputs" / "grafico_metrics_v8.png"

print(f"[v8] device = {device}")
print(f"[v8] max_iters = {max_iters}, batch = {batch_size}, block = {block_size}")


# ------------------------------------------------------------
# Corpus: livros + Q&A
# ------------------------------------------------------------
files = sorted((ROOT / "data").glob("*.txt"))
print(f"[v8] {len(files)} arquivos em {ROOT / 'data'}")

blacklist = [
    "Página", "Page",
    "Colleen Hoover", "Machado de Assis",
    "Sumário", "Capítulo", "Copyright",
    "Todos os direitos reservados",
    "# MWETOOLKIT",
]

all_text = ""
for fn in files:
    try:
        with open(fn, "r", encoding="utf-8") as f:
            raw = f.read()
        clean = []
        for line in raw.split("\n"):
            line = line.strip()
            if not line:
                continue
            if line.isdigit():
                continue
            if any(bad in line for bad in blacklist):
                continue
            if len(line) < 2 and line not in [".", "?", "!", "—"]:
                continue
            clean.append(line)
        book_text = "\n".join(clean)
        if "<|endoftext|>" not in book_text:
            book_text += " <|endoftext|>"
        all_text += book_text + "\n"
        print(f"  - {Path(fn).name}: {len(clean)} linhas úteis")
    except Exception as e:
        print(f"  ! erro em {fn}: {e}")

print(f"[v8] corpus total: {len(all_text)/1e6:.2f} MB")


# ------------------------------------------------------------
# Tokenização
# ------------------------------------------------------------
enc = tiktoken.get_encoding("gpt2")
encoded = enc.encode(all_text, allowed_special={"<|endoftext|>"})
print(f"[v8] vocab_size = {vocab_size}, total tokens = {len(encoded):,}")

data = torch.tensor(encoded, dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]
print(f"[v8] train tokens = {len(train_data):,} | val tokens = {len(val_data):,}")


# ------------------------------------------------------------
# Infra de treino (modelo vem de gpt/model.py)
# ------------------------------------------------------------
def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i:i + block_size] for i in ix])
    y = torch.stack([d[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_metrics():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_batches)
        accs = torch.zeros(eval_batches)
        for k in range(eval_batches):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
            pred = torch.argmax(F.softmax(logits, dim=-1), dim=-1)
            accs[k] = (pred == Y.view(-1)).float().mean().item()
        out[split] = {"loss": losses.mean().item(), "acc": accs.mean().item()}
    model.train()
    return out


def save_checkpoint(step, model, optimizer, lt, lv, at, av):
    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss_train": lt, "loss_val": lv,
        "acc_train": at, "acc_val": av,
    }, CHECKPOINT_FILE)


def load_checkpoint(model, optimizer):
    if not os.path.exists(CHECKPOINT_FILE):
        print(f"[v8] sem checkpoint — treino do zero")
        return 0, [], [], [], []
    print(f"[v8] carregando {CHECKPOINT_FILE}...")
    ck = torch.load(CHECKPOINT_FILE, map_location=device, weights_only=False)
    model.load_state_dict(ck["model_state_dict"])
    # Se o optimizer_state_dict for vazio (caso warm-start a partir de V7 onde
    # queremos momentums frescos), não carrega — mantém o optimizer recém-criado.
    opt_state = ck.get("optimizer_state_dict") or {}
    if opt_state.get("state") and opt_state.get("param_groups"):
        optimizer.load_state_dict(opt_state)
        print(f"[v8] optimizer carregado — retomando do step {ck['step']}")
        return ck["step"], ck.get("loss_train", []), ck.get("loss_val", []), ck.get("acc_train", []), ck.get("acc_val", [])
    else:
        print(f"[v8] apenas pesos carregados (warm-start) — treino reinicia no step 0 com LR fresco")
        return 0, [], [], [], []


# ------------------------------------------------------------
# Loop de treino
# ------------------------------------------------------------
model = GPTLanguageModel().to(device)
params_m = sum(p.numel() for p in model.parameters()) / 1e6
print(f"[v8] modelo: {params_m:.2f}M parâmetros")

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iters, eta_min=1e-5)

start_iter, lt, lv, at, av = load_checkpoint(model, optimizer)

print(f"[v8] ===== INICIANDO TREINO =====")
t0 = time.time()

best_val_loss = float("inf")
patience_counter = 0

for it in range(start_iter, max_iters):
    if it % eval_iters == 0:
        m = estimate_metrics()
        tl, vl = m["train"]["loss"], m["val"]["loss"]
        ta, va = m["train"]["acc"], m["val"]["acc"]
        elapsed = (time.time() - t0) / 60
        remaining = elapsed / max(it - start_iter, 1) * (max_iters - it) if it > start_iter else 0
        marker = ""
        if vl < best_val_loss:
            best_val_loss = vl
            patience_counter = 0
            # salva melhor checkpoint em arquivo SEPARADO (não sobrescrito)
            save_checkpoint(it, model, optimizer, lt + [tl], lv + [vl], at + [ta], av + [va])
            os.replace(CHECKPOINT_FILE, BEST_CHECKPOINT)
            marker = " ★ NOVO MELHOR"
        else:
            patience_counter += 1
            marker = f" (paciência {patience_counter}/{early_stop_patience})"
        print(f"step {it:>5} | loss {tl:.3f}/{vl:.3f} | acc {ta:.3f}/{va:.3f} | elapsed {elapsed:.1f}min | ETA {remaining:.1f}min{marker}", flush=True)
        lt.append(tl); lv.append(vl); at.append(ta); av.append(va)
        save_checkpoint(it, model, optimizer, lt, lv, at, av)  # rolling last
        if patience_counter >= early_stop_patience:
            print(f"[v8] early stopping: val não melhora há {early_stop_patience} evals. Melhor val_loss = {best_val_loss:.3f}")
            break

    xb, yb = get_batch("train")
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()

elapsed = (time.time() - t0) / 60
print(f"\n[v8] treino finalizado em {elapsed:.2f} minutos.")
# Promove best → final
if os.path.exists(BEST_CHECKPOINT):
    os.replace(BEST_CHECKPOINT, CHECKPOINT_FILE)
    print(f"[v8] melhor checkpoint promovido a {CHECKPOINT_FILE} (val_loss = {best_val_loss:.3f})")


# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------
os.makedirs("outputs", exist_ok=True)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
ax1.plot(lt, label="Treino"); ax1.plot(lv, label="Validação")
ax1.set_title("Loss (Cross-Entropy) — v8"); ax1.set_xlabel(f"Iterações (x {eval_iters})")
ax1.set_ylabel("Loss"); ax1.legend(); ax1.grid(True)
ax2.plot(at, label="Treino"); ax2.plot(av, label="Validação")
ax2.set_title("Accuracy — v8"); ax2.set_xlabel(f"Iterações (x {eval_iters})")
ax2.set_ylabel("Accuracy"); ax2.legend(); ax2.grid(True)
plt.tight_layout()
plt.savefig(PLOT_FILE)
print(f"[v8] gráfico salvo em {PLOT_FILE}")


# ------------------------------------------------------------
# Geração de exemplo — formato Q&A + continuação narrativa
# ------------------------------------------------------------
print("\n" + "=" * 50)
print("GERAÇÃO v8 — EXEMPLOS")
print("=" * 50)

model.eval()
eot_id = enc.eot_token


@torch.no_grad()
def gerar(prompt, max_new=150, temperature=0.7, top_k=50):
    ids = enc.encode(prompt) if prompt else [0]
    idx = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    for _ in range(max_new):
        idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature
        if top_k:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("inf")
        probs = F.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, nxt), dim=1)
        if nxt.item() == eot_id:
            break
    out = enc.decode(idx[0].tolist())
    return out.replace("<|endoftext|>", "").strip()


torch.manual_seed(7)
for p in [
    "Pergunta: O que é inteligência artificial?\n\nResposta:",
    "Pergunta: Quem foi Machado de Assis?\n\nResposta:",
    "Pergunta: Como fazer um bolo de chocolate?\n\nResposta:",
    "Era uma vez um menino que",
    "A velha senhora olhou pela janela e",
]:
    print("\n" + "-" * 50)
    print(gerar(p))


# ------------------------------------------------------------
# Save final model state_dict (sem optimizer)
# ------------------------------------------------------------
torch.save(model.state_dict(), FINAL_MODEL)
print(f"\n[v8] state_dict final salvo em {FINAL_MODEL}")
print("[v8] pronto.")
