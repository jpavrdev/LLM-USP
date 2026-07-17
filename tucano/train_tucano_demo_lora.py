"""LoRA SFT do Tucano-1b1-Instruct nos Q&A curados da demo (data/qa_demo_ptbr.txt).

Treina só um adapter LoRA (base congelada) em bf16, no formato de chat nativo
do modelo (o mesmo que o backend serve), com loss mascarada no turno do
assistant. Saída: tucano_sft/lora_demo/ (adapter peft).

Uso (a partir da raiz do projeto):
    webapp/.venv/bin/python tucano/train_tucano_demo_lora.py
"""
import random
import re
import time
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model

MODEL_ID = "TucanoBR/Tucano-1b1-Instruct"
ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "tucano_sft" / "lora_demo"
OUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_FILE = ROOT / "data" / "qa_demo_ptbr.txt"

MAX_LEN = 320
BATCH_SIZE = 2
GRAD_ACCUM = 2
LR = 2e-4
NUM_EPOCHS = 12
WARMUP_RATIO = 0.05
EVAL_EVERY = 40
SAVE_EVERY = 80
SEED = 42
IGNORE_INDEX = -100

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED)
random.seed(SEED)
print(f"[lora] device={device} LR={LR} batch={BATCH_SIZE}x{GRAD_ACCUM} epochs={NUM_EPOCHS}", flush=True)


def parse_qa(path: Path) -> list[tuple[str, str]]:
    text = path.read_text(encoding="utf-8")
    out = []
    for chunk in text.split("<|endoftext|>"):
        chunk = chunk.strip()
        if not chunk:
            continue
        m = re.match(r"Pergunta:\s*(.*?)\n\n+Resposta:\s*(.*)", chunk, re.DOTALL)
        if m:
            out.append((m.group(1).strip(), m.group(2).strip()))
    return out


examples = parse_qa(DATA_FILE)
random.shuffle(examples)
print(f"[lora] {len(examples)} exemplos em {DATA_FILE.name}", flush=True)

print(f"[lora] carregando {MODEL_ID}...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16).to(device)
model.gradient_checkpointing_enable()
model.enable_input_require_grads()
model.config.use_cache = False

lora_cfg = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)
model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()


class ChatSFTDataset(Dataset):
    def __init__(self, examples, tokenizer, max_len=MAX_LEN):
        self.items = []
        for q, a in examples:
            prompt_str = tokenizer.apply_chat_template(
                [{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True
            )
            full_str = tokenizer.apply_chat_template(
                [{"role": "user", "content": q}, {"role": "assistant", "content": a}],
                tokenize=False, add_generation_prompt=False,
            )
            prompt_ids = tokenizer(prompt_str, add_special_tokens=False).input_ids
            full_ids = tokenizer(full_str, add_special_tokens=False, truncation=True, max_length=max_len).input_ids
            labels = [IGNORE_INDEX] * len(prompt_ids) + full_ids[len(prompt_ids):]
            labels = labels[:len(full_ids)]
            if all(l == IGNORE_INDEX for l in labels):
                continue
            self.items.append({"input_ids": full_ids, "labels": labels})

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def collate(batch):
    pad_id = tokenizer.pad_token_id
    n = max(len(b["input_ids"]) for b in batch)
    input_ids, labels, attn = [], [], []
    for b in batch:
        k = len(b["input_ids"])
        input_ids.append(b["input_ids"] + [pad_id] * (n - k))
        labels.append(b["labels"] + [IGNORE_INDEX] * (n - k))
        attn.append([1] * k + [0] * (n - k))
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attn, dtype=torch.long),
    }


dataset = ChatSFTDataset(examples, tokenizer)
print(f"[lora] dataset pronto: {len(dataset)} exemplos tokenizados", flush=True)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)
total_steps = (len(loader) // GRAD_ACCUM) * NUM_EPOCHS
warmup_steps = max(1, int(total_steps * WARMUP_RATIO))
print(f"[lora] total_steps={total_steps} warmup={warmup_steps}", flush=True)

optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)


def sample_answer(q: str) -> str:
    model.eval()
    prompt_str = tokenizer.apply_chat_template(
        [{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True
    )
    prompt_ids = tokenizer(prompt_str, add_special_tokens=False).input_ids
    inputs = torch.tensor([prompt_ids], device=device)
    with torch.no_grad():
        out = model.generate(
            input_ids=inputs, max_new_tokens=120, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(out[0][len(prompt_ids):], skip_special_tokens=True)
    model.train()
    return text.strip()


print("\n[lora] ===== INICIANDO TREINO =====", flush=True)
t0 = time.time()
step = 0
accum = 0
running = 0.0
model.train()

for epoch in range(NUM_EPOCHS):
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        loss = out.loss / GRAD_ACCUM
        if not torch.isfinite(loss):
            accum = 0
            optimizer.zero_grad(set_to_none=True)
            print("  [aviso] loss não-finita, batch descartado", flush=True)
            continue
        loss.backward()
        running += loss.item() * GRAD_ACCUM
        accum += 1
        if accum == GRAD_ACCUM:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            step += 1
            accum = 0
            if step % 5 == 0 or step == total_steps:
                avg = running / (5 * GRAD_ACCUM) if step % 5 == 0 else running / GRAD_ACCUM
                el = (time.time() - t0) / 60
                print(f"step {step}/{total_steps} | loss {avg:.4f} | lr {scheduler.get_last_lr()[0]:.2e} | {el:.1f}min", flush=True)
                running = 0.0
            if step % EVAL_EVERY == 0:
                print(f"[amostra step {step}] 'Quem escreveu Dom Casmurro?' -> {sample_answer('Quem escreveu Dom Casmurro?')[:200]}", flush=True)
            if step % SAVE_EVERY == 0:
                ckpt = OUT_DIR / f"step_{step}"
                model.save_pretrained(ckpt)
                print(f"[ckpt] salvo em {ckpt.name}", flush=True)

model.save_pretrained(OUT_DIR)
tokenizer.save_pretrained(OUT_DIR)
print(f"\n[lora] DONE em {(time.time()-t0)/60:.1f}min. Adapter salvo em {OUT_DIR}", flush=True)

print("\n===== AMOSTRAS FINAIS =====", flush=True)
for q in ["Quem escreveu Dom Casmurro?", "Quem foi Machado de Assis?", "O que é fotossíntese?", "Qual a capital do Brasil?"]:
    print(f"\n> {q}\n{sample_answer(q)}", flush=True)
