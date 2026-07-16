"""Smoke test do Tucano-160m: baixa pesos + tokenizer e gera texto de exemplo.

Uso (a partir da raiz do projeto):
    webapp/.venv/bin/python tucano/test_tucano.py
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "TucanoBR/Tucano-160m"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[tucano] device = {device}")

print(f"[tucano] baixando/carregando {MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
).to(device)

n_params = sum(p.numel() for p in model.parameters())
print(f"[tucano] {n_params/1e6:.1f}M parâmetros carregados")
print(f"[tucano] vocab size: {tokenizer.vocab_size}")
print(f"[tucano] tokenizer class: {tokenizer.__class__.__name__}")

# Prompts do próprio card do modelo
prompts = [
    "A floresta da Amazônia é conhecida por sua",
    "Uma das coisas que Portugal, Angola, Brasil e Moçambique tem em comum é o",
    "O Carnaval do Rio de Janeiro é",
    "Era uma vez um menino que",
    "Olá! Como você está?",
]

model.eval()
for p in prompts:
    print(f"\n--- {p!r} ---")
    inputs = tokenizer(p, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            pad_token_id=tokenizer.eos_token_id,
        )
    print(tokenizer.decode(out[0], skip_special_tokens=True))

# Quanto ocupa na VRAM
if device == "cuda":
    mem_mb = torch.cuda.memory_allocated() / 1e6
    print(f"\n[tucano] VRAM ocupada: {mem_mb:.1f} MB")
