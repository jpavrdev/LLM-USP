"""Smoke test do Tucano-1b1-Instruct: download + amostras com chat template oficial.

Esperado: modelo responde diretamente sem precisar envelopar em Alpaca-style,
pois foi fine-tunado com seu próprio chat template pela PUCRS.
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "TucanoBR/Tucano-1b1-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[tucano-1b1] device = {device}")
print(f"[tucano-1b1] baixando/carregando {MODEL_ID}...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.bfloat16,  # bf16 pra caber em 6 GB
).to(device)

n_params = sum(p.numel() for p in model.parameters())
print(f"[tucano-1b1] {n_params/1e6:.1f}M params carregados")
print(f"[tucano-1b1] vocab_size: {tokenizer.vocab_size}")
print(f"[tucano-1b1] chat_template disponível: {bool(tokenizer.chat_template)}")

model.eval()

questions = [
    "Quem é o autor de Dom Casmurro?",
    "Qual a capital do Brasil?",
    "O que é a Floresta Amazônica?",
    "Liste três dicas práticas para melhorar o foco no trabalho.",
    "Explique em uma frase o que é fotossíntese.",
    "Escreva uma breve história sobre um astronauta que descobre um planeta habitado.",
    "Quem foi Machado de Assis?",
]

for q in questions:
    # Usa o chat template oficial do modelo (Llama/Tucano style)
    if tokenizer.chat_template:
        messages = [{"role": "user", "content": q}]
        # API transformers 5.x: retorna dict; em <5 retorna tensor.
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        prompt_text = f"User: {q}\nAssistant:"
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    in_len = inputs.input_ids.shape[1]

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=160,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
        )
    # Decode só a parte nova
    response = tokenizer.decode(out[0][in_len:], skip_special_tokens=True)
    print(f"\n▶ {q}")
    print(response.strip())

# VRAM final
if device == "cuda":
    mem_mb = torch.cuda.memory_allocated() / 1e6
    print(f"\n[tucano-1b1] VRAM ocupada: {mem_mb:.0f} MB")
