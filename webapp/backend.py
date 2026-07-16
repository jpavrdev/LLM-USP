"""Backend FastAPI para o Tucano-1b1-Instruct (modelo oficial PUCRS).

Usa o chat template nativo do modelo (Llama-style com papéis user/assistant).
"""
import asyncio
import json
import os
import re
from pathlib import Path

import torch
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent

# Modelo oficial PUCRS — Tucano-1b1-Instruct
MODEL_ID = os.environ.get("MODEL_ID", "TucanoBR/Tucano-1b1-Instruct")
MODEL_LABEL = "Tucano-1b1-Instruct (PUCRS, 1.1B params)"

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" else torch.float32

print(f"[backend] device = {device}, dtype = {dtype}")
print(f"[backend] carregando {MODEL_ID}...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=dtype).to(device)
model.eval()

n_params = sum(p.numel() for p in model.parameters())
print(f"[backend] modelo pronto: {n_params/1e6:.1f}M params")
print(f"[backend] chat_template: {bool(tokenizer.chat_template)}")


SYSTEM_PREFIX = (
    "Responda em português brasileiro de forma precisa e factual, em um parágrafo direto. "
    "Se não tiver certeza da resposta, admita que não sabe em vez de inventar informações.\n\n"
    "Pergunta: "
)


def build_prompt(user_message: str) -> str:
    """Monta o prompt com instrução de honestidade prependida ao turno do usuário.

    O chat template do Tucano-1b1-Instruct é rígido (só user/assistant alternando),
    então a instrução de sistema é inserida no próprio <instruction>...</instruction>.
    """
    user_content = SYSTEM_PREFIX + user_message.strip()
    messages = [{"role": "user", "content": user_content}]
    if tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return f"User: {user_content}\nAssistant:"


# Token IDs dos special tokens (muito mais robusto que string match)
INSTR_END_ID = tokenizer.convert_tokens_to_ids("</instruction>")
INSTR_START_ID = tokenizer.convert_tokens_to_ids("<instruction>")
EOS_ID = tokenizer.eos_token_id

# Special tokens que NUNCA devem sofrer repetition_penalty — incluindo EOS e
# marcadores de template, pra o modelo manter capacidade de encerrar/estruturar.
SPECIAL_TOKEN_IDS = set(tokenizer.all_special_ids) | {INSTR_END_ID, INSTR_START_ID}


INSTR_END_STR = "</instruction>"
INSTR_START_STR = "<instruction>"
NOISE_PREFIXES = ("\nResposta:", "Resposta:", "\nResposta", "Resposta")
JUNK_TAGS = ("</instruction>", "<instruction>", "<s>", "</s>", "<pad>", "<unk>")

# Se a tag aparecer até esse número de tokens, descarta o buffer (alucinação inicial)
EARLY_HALLUCINATION_LIMIT = 20


@torch.no_grad()
def generate_tokens(prompt_user: str, max_new_tokens: int, temperature: float, top_k: int | None, top_p: float | None, repetition_penalty: float = 1.1):
    """Stream token-a-token com filtragem por string (o modelo gera </instruction> como
    6 tokens separados, então detecção por token-id não funciona).

    Estratégia:
      1. Acumula tokens em `answer_ids`
      2. A cada passo, decoda tudo e procura `</instruction>` na string
      3. Se aparecer CEDO (<= EARLY_HALLUCINATION_LIMIT), descarta tudo ANTES dela
      4. Se aparecer TARDE, filtra a tag mas mantém o resto
      5. Emite o texto limpo incrementalmente ao cliente
    """
    prompt_text = build_prompt(prompt_user)
    enc = tokenizer(prompt_text, return_tensors="pt").to(device)
    attention_mask = enc.attention_mask

    answer_ids: list[int] = []
    started = False      # começamos a emitir a resposta real?
    decoded_so_far = ""  # texto já enviado ao cliente
    past_kv = None
    next_input = enc.input_ids
    tokens_generated = 0

    for _ in range(max_new_tokens):
        outputs = model(
            input_ids=next_input,
            attention_mask=attention_mask,
            past_key_values=past_kv,
            use_cache=True,
        )
        logits = outputs.logits[:, -1, :] / max(temperature, 1e-5)

        # Repetition penalty só em tokens já gerados, excluindo special tokens
        if repetition_penalty and repetition_penalty != 1.0:
            seen_ids = set(answer_ids) - SPECIAL_TOKEN_IDS
            for tid in seen_ids:
                if logits[0, tid] > 0:
                    logits[0, tid] = logits[0, tid] / repetition_penalty
                else:
                    logits[0, tid] = logits[0, tid] * repetition_penalty

        if top_k is not None and top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("inf")

        if top_p is not None and 0 < top_p < 1:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            cumulative = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            remove = cumulative > top_p
            remove[:, 1:] = remove[:, :-1].clone()
            remove[:, 0] = False
            indices_to_remove = remove.scatter(1, sorted_idx, remove)
            logits[indices_to_remove] = -float("inf")

        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        tok = int(next_id.item())

        if tok == EOS_ID:
            break

        past_kv = outputs.past_key_values
        next_input = next_id
        attention_mask = torch.cat([attention_mask, torch.ones_like(next_id)], dim=1)
        tokens_generated += 1
        answer_ids.append(tok)

        # Decoda tudo com tags especiais visíveis (pra pegar </instruction> literal)
        raw = tokenizer.decode(answer_ids, skip_special_tokens=False)

        # Fase 1: ainda não começou a resposta real
        if not started:
            if INSTR_END_STR in raw:
                # Alucinação de multi-turn: modelo completou o prompt com </instruction>
                # Descarta tudo ATÉ a tag, começa a emitir o que vem depois
                idx = raw.rfind(INSTR_END_STR) + len(INSTR_END_STR)
                cleaned = raw[idx:]
                # Remove outras tags residuais
                for tag in JUNK_TAGS:
                    cleaned = cleaned.replace(tag, "")
                # Remove prefixos de "Resposta:" que o modelo às vezes deixa
                for p in NOISE_PREFIXES:
                    if cleaned.startswith(p):
                        cleaned = cleaned[len(p):].lstrip(":").lstrip()
                        break
                cleaned = cleaned.lstrip()
                started = True
                if cleaned:
                    decoded_so_far = cleaned
                    yield cleaned
                continue
            if tokens_generated > EARLY_HALLUCINATION_LIMIT:
                # Nenhum marcador apareceu — assume resposta direta
                cleaned = raw
                for tag in JUNK_TAGS:
                    cleaned = cleaned.replace(tag, "")
                for p in NOISE_PREFIXES:
                    if cleaned.startswith(p):
                        cleaned = cleaned[len(p):].lstrip(":").lstrip()
                        break
                cleaned = cleaned.lstrip()
                started = True
                if cleaned:
                    decoded_so_far = cleaned
                    yield cleaned
                continue
            # Bufferiza sem emitir
            continue

        # Fase 2: já começou a emitir resposta real
        # Se modelo tenta abrir novo turno <instruction>..., encerra
        if INSTR_START_STR in raw:
            break

        # Limpa tags que possam ter vazado e emite delta
        cleaned = raw
        # Pega só o trecho após o último </instruction> (se houver)
        if INSTR_END_STR in cleaned:
            idx = cleaned.rfind(INSTR_END_STR) + len(INSTR_END_STR)
            cleaned = cleaned[idx:]
        for tag in JUNK_TAGS:
            cleaned = cleaned.replace(tag, "")
        # No meio da resposta, não re-limpa prefixos "Resposta:" — eles podem
        # ser parte legítima do conteúdo
        delta = cleaned[len(decoded_so_far):]
        if delta:
            decoded_so_far = cleaned
            yield delta


app = FastAPI(title=MODEL_LABEL)


class ChatRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 250
    temperature: float = 0.4
    top_k: int | None = 50
    top_p: float | None = 0.9
    repetition_penalty: float = 1.1


@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/api/info")
def info():
    return {
        "device": device,
        "dtype": str(dtype).replace("torch.", ""),
        "params_millions": round(n_params / 1e6, 2),
        "model": MODEL_LABEL,
        "model_id": MODEL_ID,
        "vocab_size": tokenizer.vocab_size,
        "max_context": getattr(model.config, "max_position_embeddings", None),
        "is_chat": True,
    }


@app.post("/api/chat")
async def chat(req: ChatRequest):
    async def event_stream():
        loop = asyncio.get_running_loop()
        gen = generate_tokens(req.prompt, req.max_new_tokens, req.temperature, req.top_k, req.top_p, req.repetition_penalty)
        try:
            while True:
                delta = await loop.run_in_executor(None, next, gen, None)
                if delta is None:
                    break
                yield f"data: {json.dumps({'delta': delta})}\n\n"
            yield "data: [DONE]\n\n"
        except asyncio.CancelledError:
            raise

    return StreamingResponse(event_stream(), media_type="text/event-stream")


app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")
