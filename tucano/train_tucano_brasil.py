"""SFT adicional sobre tucano_sft/final, focado em conhecimento brasileiro.

Mix:
  - 162 pares Q&A brasileiros (replicados 3x = 486) — reforço do conhecimento novo
  - 500 pares aleatórios do Dolly/Alpaca — anti-esquecimento do chat geral

LR baixo (5e-6) e poucos steps pra não desestabilizar o modelo já fine-tunado.
Saída: ../tucano_sft/final_brasil/

Uso (a partir da raiz do projeto):
    webapp/.venv/bin/python tucano/train_tucano_brasil.py
"""
import os
import random
import re
import time
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_cosine_schedule_with_warmup,
)


BASE_MODEL = Path(__file__).resolve().parent.parent / "tucano_sft" / "final"
OUT_DIR = Path(__file__).resolve().parent.parent / "tucano_sft" / "final_brasil"
OUT_DIR.mkdir(exist_ok=True, parents=True)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Hiperparâmetros (LR baixo + poucos steps = fine-tune suave)
MAX_LEN = 512
BATCH_SIZE = 1
GRAD_ACCUM = 16
LR = 5e-6
NUM_EPOCHS = 3
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
EVAL_EVERY = 50       # amostras a cada 50 steps
SEED = 42
BRASIL_REPEAT = 3     # replica pares brasileiros pra reforçar
ANTI_FORGET_SAMPLES = 500  # pares aleatórios de Dolly/Alpaca misturados


device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED)
random.seed(SEED)
print(f"[sft-brasil] device={device}, base={BASE_MODEL.name}, LR={LR}, epochs={NUM_EPOCHS}")


# ---------------------------------------------------------------
# Carrega dados
# ---------------------------------------------------------------
def parse_qa_file(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    examples = []
    for chunk in text.split("<|endoftext|>"):
        chunk = chunk.strip()
        if not chunk:
            continue
        m = re.match(r"Pergunta:\s*(.*?)\n\n+Resposta:\s*(.*)", chunk, re.DOTALL)
        if not m:
            continue
        instr = m.group(1).strip()
        resp = m.group(2).strip()
        if len(instr) < 5 or len(resp) < 5:
            continue
        examples.append({"instruction": instr, "response": resp})
    return examples


print("\n[sft-brasil] carregando datasets...")
brasil = parse_qa_file(DATA_DIR / "qa_brasil_ptbr.txt")
dolly = parse_qa_file(DATA_DIR / "qa_dolly_ptbr.txt")
alpaca = parse_qa_file(DATA_DIR / "qa_alpaca_ptbr.txt")
print(f"  brasil: {len(brasil)} pares")
print(f"  dolly:  {len(dolly)} pares")
print(f"  alpaca: {len(alpaca)} pares")

# Amostra anti-forget do Dolly+Alpaca
antishock = dolly + alpaca
random.shuffle(antishock)
antishock = antishock[:ANTI_FORGET_SAMPLES]

# Dataset final: brasileiros replicados + anti-forget
all_examples = brasil * BRASIL_REPEAT + antishock
random.shuffle(all_examples)
print(f"\n[sft-brasil] mix de treino: {len(all_examples)} exemplos "
      f"({len(brasil)*BRASIL_REPEAT} brasil × {BRASIL_REPEAT} + {len(antishock)} anti-forget)")


# ---------------------------------------------------------------
# Modelo e tokenizer — carrega do fine-tune atual
# ---------------------------------------------------------------
print(f"\n[sft-brasil] carregando modelo de {BASE_MODEL}...")
tokenizer = AutoTokenizer.from_pretrained(str(BASE_MODEL))
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(str(BASE_MODEL), dtype=torch.float32)
model.gradient_checkpointing_enable()
model.to(device)
print(f"[sft-brasil] modelo: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")


# ---------------------------------------------------------------
# Dataset / collator (mesmo padrão do treino anterior)
# ---------------------------------------------------------------
PROMPT_TEMPLATE = "### Instrução:\n{instruction}\n\n### Resposta:\n"
IGNORE_INDEX = -100


class SFTDataset(Dataset):
    def __init__(self, examples, tokenizer, max_len=MAX_LEN):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.examples = []
        for ex in examples:
            prompt = PROMPT_TEMPLATE.format(instruction=ex["instruction"])
            prompt_len = len(tokenizer(prompt, add_special_tokens=False).input_ids)
            if prompt_len + 1 + 16 > max_len:
                continue
            self.examples.append(ex)

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


dataset = SFTDataset(all_examples, tokenizer)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)
total_steps = (len(loader) // GRAD_ACCUM) * NUM_EPOCHS
warmup_steps = max(1, int(total_steps * WARMUP_RATIO))
print(f"[sft-brasil] steps totais={total_steps}, warmup={warmup_steps}")


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
# Loop
# ---------------------------------------------------------------
print("\n[sft-brasil] ===== TREINANDO =====")
t0 = time.time()
step = accum = 0
running_loss = 0.0
model.train()

test_prompts = [
    "Quem é o autor de Dom Casmurro?",
    "Qual a capital do Brasil?",
    "Quem foi Machado de Assis?",
]

for epoch in range(NUM_EPOCHS):
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        loss = out.loss / GRAD_ACCUM
        if not torch.isfinite(loss):
            accum = 0; optimizer.zero_grad(set_to_none=True)
            print("  [aviso] loss não-finita, pulando batch", flush=True)
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

            if step % 10 == 0:
                avg = running_loss / (10 * GRAD_ACCUM)
                elapsed = (time.time() - t0) / 60
                print(f"step {step}/{total_steps} | loss {avg:.3f} | lr {scheduler.get_last_lr()[0]:.2e} | {elapsed:.1f}min", flush=True)
                running_loss = 0.0

            if step % EVAL_EVERY == 0:
                model.eval()
                for tp in test_prompts:
                    p_full = PROMPT_TEMPLATE.format(instruction=tp)
                    inputs = tokenizer(p_full, return_tensors="pt").to(device)
                    with torch.no_grad():
                        out_ids = model.generate(**inputs, max_new_tokens=80, do_sample=False, pad_token_id=tokenizer.eos_token_id)
                    sample = tokenizer.decode(out_ids[0], skip_special_tokens=True)
                    resp = sample[len(p_full):].split("###")[0].strip()[:200]
                    print(f"  [step {step}] Q: {tp}")
                    print(f"  [step {step}] A: {resp}", flush=True)
                model.train()


elapsed = (time.time() - t0) / 60
print(f"\n[sft-brasil] treino finalizado em {elapsed:.2f} min")

model.save_pretrained(str(OUT_DIR))
tokenizer.save_pretrained(str(OUT_DIR))
print(f"[sft-brasil] modelo salvo em {OUT_DIR}")

# Amostras finais
print("\n" + "=" * 50)
print("AMOSTRAS FINAIS (após SFT Brasil)")
print("=" * 50)
model.eval()
test_set = [
    "Quem é o autor de Dom Casmurro?",
    "Quem escreveu Grande Sertão: Veredas?",
    "Qual a capital do Brasil?",
    "Qual é a maior cidade do Brasil?",
    "Quando foi a Proclamação da República?",
    "O que é a Floresta Amazônica?",
    "Fale sobre a Bossa Nova.",
    "Quem foi Pelé?",
    "Liste três dicas práticas para melhorar o foco no trabalho.",
]
for q in test_set:
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
