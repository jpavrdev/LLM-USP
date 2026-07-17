"""Verificador da demo: roda as perguntas curadas no modelo + adapter LoRA e
confere se os fatos-chave aparecem na resposta (greedy, reproduzível).

Rode após qualquer retriino/ajuste pra acompanhar a qualidade ao longo do tempo:
    webapp/.venv/bin/python tucano/verify_demo.py
"""
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

ROOT = Path(__file__).resolve().parent.parent
MODEL_ID = "TucanoBR/Tucano-1b1-Instruct"
ADAPTER = ROOT / "tucano_sft" / "lora_demo"

# (pergunta, [fatos-chave esperados na resposta])
CHECKS = [
    ("Quem foi Machado de Assis?", ["1839", "1908", "Academia Brasileira de Letras"]),
    ("Quem escreveu Dom Casmurro?", ["Machado de Assis", "1899", "Dom Casmurro"]),
    ("Quem escreveu Memórias Póstumas de Brás Cubas?", ["Machado de Assis", "1881"]),
    ("Quem foi o primeiro presidente da Academia Brasileira de Letras?", ["Machado de Assis"]),
    ("O que é inteligência artificial?", ["intelig", "aprendizado"]),
    ("O que é aprendizado de máquina?", ["dados", "aprendizado"]),
    ("O que é fotossíntese?", ["luz solar", "oxigênio"]),
    ("Qual a capital do Brasil?", ["Brasília", "1960", "Niemeyer"]),
    ("O que foi o Realismo no Brasil?", ["Machado"]),
]

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[verify] device={device}", flush=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16).to(device)
if (ADAPTER / "adapter_config.json").exists():
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, str(ADAPTER))
    print(f"[verify] adapter LoRA carregado", flush=True)
model.eval()


def answer(q: str) -> str:
    prompt = tokenizer.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)
    ids = torch.tensor([tokenizer(prompt, add_special_tokens=False).input_ids], device=device)
    with torch.no_grad():
        out = model.generate(input_ids=ids, max_new_tokens=130, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()


print("\n===== VERIFICAÇÃO DA DEMO =====\n", flush=True)
ok = 0
for q, facts in CHECKS:
    a = answer(q)
    missing = [f for f in facts if f.lower() not in a.lower()]
    status = "✅" if not missing else "⚠️"
    if not missing:
        ok += 1
    print(f"{status} {q}", flush=True)
    print(f"   {a[:200]}", flush=True)
    if missing:
        print(f"   faltando: {missing}", flush=True)
    print(flush=True)

print(f"===== {ok}/{len(CHECKS)} perguntas com todos os fatos-chave =====", flush=True)
sys.exit(0 if ok == len(CHECKS) else 1)
