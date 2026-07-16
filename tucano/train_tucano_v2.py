"""SFT do Tucano-160m base com o dataset oficial TucanoBR/Tucano-SFT.

Estratégia: do zero (a partir do Tucano-160m base, não do nosso fine-tune anterior).
  - Dataset: 679k exemplos oficiais, formato [{role:'user'/'assistant', content}]
  - Extração: primeiro user → instrução, primeiro assistant → resposta
  - Template: mesmo Alpaca-style do v1 (### Instrução:/### Resposta:)
  - Subset: 100k exemplos aleatórios (1 epoch ~2h30min na GPU)

Saída: tucano_sft/v2_tucanoSFT/
"""
import os
import random
import re
import time
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup
from datasets import load_dataset


MODEL_ID = "TucanoBR/Tucano-160m"
OUT_DIR = Path(__file__).resolve().parent.parent / "tucano_sft" / "v2_tucanoSFT"
OUT_DIR.mkdir(exist_ok=True, parents=True)

# Hiperparâmetros
MAX_LEN = 512
BATCH_SIZE = 1
GRAD_ACCUM = 16
LR = 1e-5
NUM_EPOCHS = 1
WARMUP_RATIO = 0.03
WEIGHT_DECAY = 0.01
EVAL_EVERY = 500
SAMPLE_SIZE = 100_000   # de 679k, usar 100k
SEED = 42


device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED); random.seed(SEED)
print(f"[v2] device={device}, base={MODEL_ID}, LR={LR}, sample={SAMPLE_SIZE}, epochs={NUM_EPOCHS}")


# ---------------------------------------------------------------
# Carrega dataset oficial
# ---------------------------------------------------------------
print("\n[v2] carregando TucanoBR/Tucano-SFT...")
ds = load_dataset("TucanoBR/Tucano-SFT", split="train")
print(f"[v2] dataset tem {len(ds):,} conversações")

# Amostra aleatória
indices = list(range(len(ds)))
random.shuffle(indices)
indices = indices[:SAMPLE_SIZE]
print(f"[v2] subset: {len(indices):,} exemplos")


# ---------------------------------------------------------------
# Tokenizer e modelo (base)
# ---------------------------------------------------------------
print(f"\n[v2] carregando {MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
model.gradient_checkpointing_enable()
model.to(device)
print(f"[v2] modelo: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")


# ---------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------
PROMPT_TEMPLATE = "### Instrução:\n{instruction}\n\n### Resposta:\n"
IGNORE_INDEX = -100


class TucanoSFTDataset(Dataset):
    def __init__(self, hf_ds, indices, tokenizer, max_len=MAX_LEN):
        self.hf_ds = hf_ds
        self.tokenizer = tokenizer
        self.max_len = max_len
        # Pré-filtro: extrai (instr, resp) e descarta os que não cabem
        self.examples = []
        skipped = 0
        for i in indices:
            conv = hf_ds[i].get("conversations", [])
            # Primeiro user, primeiro assistant
            user_msg = None
            asst_msg = None
            for msg in conv:
                if msg["role"] == "user" and user_msg is None:
                    user_msg = msg["content"].strip()
                elif msg["role"] == "assistant" and asst_msg is None and user_msg is not None:
                    asst_msg = msg["content"].strip()
                    break
            if not user_msg or not asst_msg or len(user_msg) < 3 or len(asst_msg) < 3:
                skipped += 1
                continue
            # Filtro de tamanho: prompt não pode ocupar quase todo o max_len
            prompt = PROMPT_TEMPLATE.format(instruction=user_msg)
            plen = len(tokenizer(prompt, add_special_tokens=False).input_ids)
            if plen + 16 > max_len:
                skipped += 1
                continue
            self.examples.append({"instruction": user_msg, "response": asst_msg})
        print(f"[dataset] {len(self.examples):,} exemplos válidos | {skipped:,} filtrados")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        prompt = PROMPT_TEMPLATE.format(instruction=ex["instruction"])
        response = ex["response"] + self.tokenizer.eos_token
        full = prompt + response

        prompt_ids = self.tokenizer(prompt, add_special_tokens=False).input_ids
        full_ids = self.tokenizer(full, add_special_tokens=False, truncation=True, max_length=self.max_len).input_ids

        bos = self.tokenizer.bos_token_id
        if bos is not None:
            full_ids = [bos] + full_ids
            prompt_ids = [bos] + prompt_ids

        labels = list(full_ids)
        prompt_len = min(len(prompt_ids), len(full_ids))
        for i in range(prompt_len):
            labels[i] = IGNORE_INDEX
        if all(l == IGNORE_INDEX for l in labels):
            labels[-1] = full_ids[-1]
        return {"input_ids": full_ids, "labels": labels}


def collate(batch):
    pad_id = tokenizer.pad_token_id
    max_len = max(len(b["input_ids"]) for b in batch)
    input_ids, labels, attention_mask = [], [], []
    for b in batch:
        n = len(b["input_ids"])
        input_ids.append(b["input_ids"] + [pad_id] * (max_len - n))
        labels.append(b["labels"] + [IGNORE_INDEX] * (max_len - n))
        attention_mask.append([1] * n + [0] * (max_len - n))
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
    }


print("\n[v2] preparando dataset...")
dataset = TucanoSFTDataset(ds, indices, tokenizer)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)
total_steps = (len(loader) // GRAD_ACCUM) * NUM_EPOCHS
warmup_steps = max(1, int(total_steps * WARMUP_RATIO))
print(f"[v2] steps totais={total_steps}, warmup={warmup_steps}")


# ---------------------------------------------------------------
# Otimização
# ---------------------------------------------------------------
no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight", "norm.weight"]
params_grouped = [
    {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": WEIGHT_DECAY},
    {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
]
optimizer = torch.optim.AdamW(params_grouped, lr=LR)
scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)


# ---------------------------------------------------------------
# Treino
# ---------------------------------------------------------------
print(f"\n[v2] ===== TREINANDO =====")
t0 = time.time()
step = accum = 0
running_loss = 0.0
model.train()

for epoch in range(NUM_EPOCHS):
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        loss = out.loss / GRAD_ACCUM
        if not torch.isfinite(loss):
            accum = 0; optimizer.zero_grad(set_to_none=True)
            continue
        loss.backward()
        running_loss += loss.item() * GRAD_ACCUM
        accum += 1

        if accum == GRAD_ACCUM:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            step += 1
            accum = 0

            if step % 20 == 0:
                avg = running_loss / (20 * GRAD_ACCUM)
                elapsed = (time.time() - t0) / 60
                remaining = elapsed / step * (total_steps - step) if step else 0
                print(f"step {step}/{total_steps} | loss {avg:.3f} | lr {scheduler.get_last_lr()[0]:.2e} | {elapsed:.1f}min | ETA {remaining:.0f}min", flush=True)
                running_loss = 0.0

            if step % EVAL_EVERY == 0:
                model.eval()
                for tp in ["Quem é o autor de Dom Casmurro?", "Qual a capital do Brasil?", "Liste três dicas práticas para foco no trabalho."]:
                    p_full = PROMPT_TEMPLATE.format(instruction=tp)
                    inputs = tokenizer(p_full, return_tensors="pt").to(device)
                    with torch.no_grad():
                        out_ids = model.generate(**inputs, max_new_tokens=80, do_sample=False, pad_token_id=tokenizer.eos_token_id)
                    sample = tokenizer.decode(out_ids[0], skip_special_tokens=True)
                    resp = sample[len(p_full):].split("###")[0].strip()[:200]
                    print(f"  [step {step}] Q: {tp}")
                    print(f"  [step {step}] A: {resp}", flush=True)
                # Checkpoint
                ckpt_dir = OUT_DIR / f"step_{step}"
                ckpt_dir.mkdir(exist_ok=True)
                model.save_pretrained(ckpt_dir)
                print(f"  [ckpt] salvo em step_{step}", flush=True)
                model.train()


elapsed = (time.time() - t0) / 60
print(f"\n[v2] treino finalizado em {elapsed:.2f} min")
model.save_pretrained(str(OUT_DIR))
tokenizer.save_pretrained(str(OUT_DIR))
print(f"[v2] modelo salvo em {OUT_DIR}")


# ---------------------------------------------------------------
# Amostras finais
# ---------------------------------------------------------------
print("\n" + "=" * 70)
print("AMOSTRAS FINAIS (após SFT com Tucano-SFT oficial)")
print("=" * 70)
model.eval()
for q in [
    "Quem é o autor de Dom Casmurro?",
    "Qual a capital do Brasil?",
    "O que é a Floresta Amazônica?",
    "Liste três dicas práticas para melhorar o foco no trabalho.",
    "Escreva uma breve história sobre um astronauta que descobre um planeta habitado.",
    "Explique em uma frase o que é fotossíntese.",
    "Quem foi Machado de Assis?",
]:
    p_full = PROMPT_TEMPLATE.format(instruction=q)
    inputs = tokenizer(p_full, return_tensors="pt").to(device)
    with torch.no_grad():
        out_ids = model.generate(
            **inputs, max_new_tokens=140, do_sample=True,
            temperature=0.7, top_k=50, pad_token_id=tokenizer.eos_token_id,
        )
    full = tokenizer.decode(out_ids[0], skip_special_tokens=True)
    resp = full[len(p_full):].split("###")[0].strip()
    print(f"\n▶ {q}")
    print(resp)
