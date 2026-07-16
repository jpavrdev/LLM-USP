"""SFT do Tucano-160m com os datasets Dolly+Alpaca já em data/.

Usa:
  - Formato Alpaca-style: "### Instrução:\n{q}\n\n### Resposta:\n{r}{eos}"
  - Loss mascarada só nos tokens de resposta (prompt tokens ignorados no cálculo)
  - fp16 + gradient checkpointing pra caber em 6GB VRAM
  - batch 1 x grad_accum 16 = batch efetivo 16

Uso (a partir da raiz do projeto):
    webapp/.venv/bin/python tucano/train_tucano_sft.py
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

MODEL_ID = "TucanoBR/Tucano-160m"
OUT_DIR = Path(__file__).resolve().parent.parent / "tucano_sft"
OUT_DIR.mkdir(exist_ok=True)

# Hiperparâmetros de SFT (Opção A: dataset completo, 2 epochs)
MAX_LEN = 512
BATCH_SIZE = 1
GRAD_ACCUM = 16          # batch efetivo 16
LR = 1e-5
NUM_EPOCHS = 2
WARMUP_RATIO = 0.03      # com mais steps, warmup proporcional menor
WEIGHT_DECAY = 0.01
EVAL_EVERY = 500         # a cada 500 steps (total ~5500 steps, ~11 amostras)
SAMPLE_SIZE = None       # None = usar TODOS os exemplos (~44k após filtro)
SEED = 42
USE_AMP = False

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED)
random.seed(SEED)
print(f"[sft] device = {device}")
print(f"[sft] LR={LR}, batch={BATCH_SIZE}, grad_accum={GRAD_ACCUM}, epochs={NUM_EPOCHS}, sample={SAMPLE_SIZE}")


# ---------------------------------------------------------------
# Parse dos arquivos Dolly+Alpaca já gerados pelo fetch_datasets.py
# Cada exemplo tem estrutura:
#   Pergunta: <instr>
#   <contexto opcional>
#
#   Resposta: <resp>
#   <|endoftext|>
# ---------------------------------------------------------------
def parse_qa_file(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    examples = []
    # Divide por <|endoftext|>
    for chunk in text.split("<|endoftext|>"):
        chunk = chunk.strip()
        if not chunk:
            continue
        # Separa Pergunta / Resposta
        m = re.match(r"Pergunta:\s*(.*?)\n\n+Resposta:\s*(.*)", chunk, re.DOTALL)
        if not m:
            continue
        instruction = m.group(1).strip()
        response = m.group(2).strip()
        # Filtra exemplos muito curtos (provavelmente ruído)
        if len(instruction) < 5 or len(response) < 5:
            continue
        examples.append({"instruction": instruction, "response": response})
    return examples


print("\n[sft] carregando datasets...")
data_dir = Path(__file__).resolve().parent.parent / "data"
dolly = parse_qa_file(data_dir / "qa_dolly_ptbr.txt")
alpaca = parse_qa_file(data_dir / "qa_alpaca_ptbr.txt")
print(f"  dolly  : {len(dolly):,} exemplos")
print(f"  alpaca : {len(alpaca):,} exemplos")

all_examples = dolly + alpaca
random.shuffle(all_examples)
if SAMPLE_SIZE is not None:
    all_examples = all_examples[:SAMPLE_SIZE]
    print(f"[sft] amostra selecionada: {len(all_examples)} exemplos")
else:
    print(f"[sft] usando TODOS os exemplos: {len(all_examples)}")

# ---------------------------------------------------------------
# Carregar tokenizer e modelo
# ---------------------------------------------------------------
print(f"\n[sft] carregando {MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
# O Llama tokenizer não tem pad_token por padrão; usamos eos como pad.
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
model.gradient_checkpointing_enable()  # reduz memória de ativações
model.to(device)
print(f"[sft] modelo carregado: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")


# ---------------------------------------------------------------
# Dataset com loss masking
# ---------------------------------------------------------------
PROMPT_TEMPLATE = "### Instrução:\n{instruction}\n\n### Resposta:\n"
IGNORE_INDEX = -100  # padrão PyTorch pra ignorar loss


class SFTDataset(Dataset):
    def __init__(self, examples, tokenizer, max_len=MAX_LEN):
        self.tokenizer = tokenizer
        self.max_len = max_len
        # Filtra exemplos onde o prompt já ocupa quase todo max_len — senão a
        # resposta é truncada a zero tokens e TODOS os labels viram -100 (→ NaN).
        # Pedimos pelo menos 16 tokens de resposta pra haver sinal de loss.
        self.examples = []
        skipped = 0
        for ex in examples:
            prompt = PROMPT_TEMPLATE.format(instruction=ex["instruction"])
            prompt_len = len(tokenizer(prompt, add_special_tokens=False).input_ids)
            # +1 pro BOS, e precisamos de ao menos 16 tokens livres pra resposta
            if prompt_len + 1 + 16 > max_len:
                skipped += 1
                continue
            self.examples.append(ex)
        if skipped:
            print(f"[dataset] filtrados {skipped} exemplos com prompt longo demais (>{max_len-17} tokens)")

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

        # Labels: -100 nos tokens do prompt, token_id nos tokens da resposta
        labels = list(full_ids)
        prompt_len = min(len(prompt_ids), len(full_ids))
        for i in range(prompt_len):
            labels[i] = IGNORE_INDEX

        # Salvaguarda: se por alguma razão TODOS os labels ficaram -100,
        # descarta deixando pelo menos o último token como supervisão.
        if all(l == IGNORE_INDEX for l in labels):
            labels[-1] = full_ids[-1]

        return {
            "input_ids": full_ids,
            "labels": labels,
        }


def collate(batch):
    # Pad até o maior do batch
    pad_id = tokenizer.pad_token_id
    max_len = max(len(b["input_ids"]) for b in batch)
    input_ids = []
    labels = []
    attention_mask = []
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
print(f"[sft] total steps = {total_steps}, warmup steps = {warmup_steps}")


# ---------------------------------------------------------------
# Otimizador + scheduler
# ---------------------------------------------------------------
# AdamW com weight decay em tudo exceto bias/LayerNorm
no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight", "norm.weight"]
params_grouped = [
    {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": WEIGHT_DECAY},
    {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
]
optimizer = torch.optim.AdamW(params_grouped, lr=LR)
scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

# Mixed precision — desabilitado (fp16 gerou NaN em Llama; fp32 é estável)
scaler = None


# ---------------------------------------------------------------
# Loop de treino
# ---------------------------------------------------------------
print(f"\n[sft] ===== INICIANDO SFT =====")
t0 = time.time()
step = 0
accum = 0
running_loss = 0.0
model.train()

for epoch in range(NUM_EPOCHS):
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}

        # fp32 puro — sem autocast
        out = model(**batch)
        loss = out.loss / GRAD_ACCUM

        # Salvaguarda: se a loss deu NaN (ex: batch com todos labels=-100),
        # pula o batch em vez de corromper os pesos.
        if not torch.isfinite(loss):
            accum = 0
            optimizer.zero_grad(set_to_none=True)
            print(f"  [aviso] batch descartado — loss não-finita", flush=True)
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
                avg_loss = running_loss / (20 * GRAD_ACCUM)
                elapsed = (time.time() - t0) / 60
                lr_now = scheduler.get_last_lr()[0]
                print(f"step {step:>4}/{total_steps} | loss {avg_loss:.4f} | lr {lr_now:.2e} | elapsed {elapsed:.1f}min", flush=True)
                running_loss = 0.0

            if step % EVAL_EVERY == 0:
                # Sample generation
                model.eval()
                test_prompt = "### Instrução:\nO que é inteligência artificial?\n\n### Resposta:\n"
                inputs = tokenizer(test_prompt, return_tensors="pt").to(device)
                with torch.no_grad():
                    out_ids = model.generate(**inputs, max_new_tokens=80, do_sample=False, pad_token_id=tokenizer.eos_token_id)
                sample = tokenizer.decode(out_ids[0], skip_special_tokens=True)
                print(f"[amostra step {step}] {sample[len(test_prompt):200]}...")
                model.train()

                # Save checkpoint intermediário
                ckpt_dir = OUT_DIR / f"step_{step}"
                ckpt_dir.mkdir(exist_ok=True)
                model.save_pretrained(ckpt_dir)
                tokenizer.save_pretrained(ckpt_dir)
                print(f"[ckpt] salvo em {ckpt_dir.name}")

elapsed = (time.time() - t0) / 60
print(f"\n[sft] treino finalizado em {elapsed:.2f} min")

# Salvar modelo final
final_dir = OUT_DIR / "final"
final_dir.mkdir(exist_ok=True)
model.save_pretrained(final_dir)
tokenizer.save_pretrained(final_dir)
print(f"[sft] modelo final salvo em {final_dir}")


# ---------------------------------------------------------------
# Amostras qualitativas finais
# ---------------------------------------------------------------
print("\n" + "=" * 50)
print("AMOSTRAS FINAIS")
print("=" * 50)
model.eval()
test_prompts = [
    "O que é inteligência artificial?",
    "Quem foi Machado de Assis?",
    "Como fazer um bolo de chocolate?",
    "Explique em uma frase o que é fotossíntese.",
    "Liste três autores brasileiros famosos.",
]
for q in test_prompts:
    prompt = f"### Instrução:\n{q}\n\n### Resposta:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out_ids = model.generate(
            **inputs, max_new_tokens=150, do_sample=True,
            temperature=0.7, top_k=50, pad_token_id=tokenizer.eos_token_id,
        )
    full = tokenizer.decode(out_ids[0], skip_special_tokens=True)
    print(f"\n▶ {q}")
    print(full[len(prompt):])
