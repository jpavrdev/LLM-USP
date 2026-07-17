"""Backend FastAPI para o Tucano-1b1-Instruct (modelo oficial PUCRS).

Usa o chat template nativo do modelo (Llama-style com papéis user/assistant).
"""
import asyncio
import json
import os
import sys
from pathlib import Path
from threading import Thread

import torch
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer


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

HAS_ADAPTER = False
ADAPTER_PATH = os.environ.get("ADAPTER_PATH", str(PROJECT_ROOT / "tucano_sft" / "lora_demo"))
if os.path.isdir(ADAPTER_PATH) and os.path.exists(os.path.join(ADAPTER_PATH, "adapter_config.json")):
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    HAS_ADAPTER = True
    MODEL_LABEL += " + LoRA (demo)"
    print(f"[backend] adapter LoRA carregado: {ADAPTER_PATH}")

n_params = sum(p.numel() for p in model.parameters())
print(f"[backend] modelo pronto: {n_params/1e6:.1f}M params, adapter={HAS_ADAPTER}")
print(f"[backend] chat_template: {bool(tokenizer.chat_template)}")

# RAG opcional (índice FAISS + e5 na CPU). Sem índice, segue sem RAG.
sys.path.insert(0, str(PROJECT_ROOT / "rag"))
retriever = None
try:
    from retrieve import Retriever
    if (PROJECT_ROOT / "rag" / "wiki_pt.faiss").exists():
        retriever = Retriever()
    else:
        print("[backend] índice RAG ausente, rodando sem RAG")
except Exception as e:
    print(f"[backend] RAG desabilitado: {e}")

MAX_CONTEXT_CHARS = 1800

# Prompts que pedem geração/criação (não fato): RAG só atrapalha aqui.
GENERATIVE_STARTS = (
    "escreva", "escreve", "escrever", "crie", "cria", "criar", "invente",
    "liste", "listar", "redija", "narre", "imagine", "poema",
)


def _looks_generative(prompt: str) -> bool:
    return prompt.lower().lstrip().startswith(GENERATIVE_STARTS)


SYSTEM_PREFIX = (
    "Responda em português brasileiro de forma precisa e factual, em um parágrafo direto. "
    "Se não tiver certeza da resposta, admita que não sabe em vez de inventar informações.\n\n"
    "Pergunta: "
)


def build_prompt(user_message: str, context=None) -> str:
    user_message = user_message.strip()
    if HAS_ADAPTER:
        user_content = user_message
    elif context:
        ctx = "\n\n".join(f"{h['title']}: {h['text']}" for h in context)[:MAX_CONTEXT_CHARS]
        user_content = (
            f"Contexto:\n{ctx}\n\n"
            "Responda à pergunta em português brasileiro, de forma curta e direta, "
            "com base no contexto acima.\n\n"
            f"Pergunta: {user_message}"
        )
    else:
        user_content = SYSTEM_PREFIX + user_message
    messages = [{"role": "user", "content": user_content}]
    if tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return f"User: {user_content}\nAssistant:"


INSTR_END_STR = "</instruction>"
INSTR_START_STR = "<instruction>"
EOS_STR = "</s>"
JUNK_TAGS = ("</instruction>", "<instruction>", "<s>", "</s>", "<pad>", "<unk>")
STOP_STRINGS = ("<instruction", "Resposta correta", "Resposta incorreta", "Opções")
CUT_MARKERS = ("<instruction", "</instruction", "Resposta:", "Resposta correta", "Resposta incorreta", "Opções")
NO_REPEAT_NGRAM = 3
START_BUFFER = 80

_TEMPLATE_TAGS = (INSTR_END_STR, INSTR_START_STR, EOS_STR)


def _partial_tag_len(text: str) -> int:
    best = 0
    for tag in _TEMPLATE_TAGS:
        limit = min(len(tag) - 1, len(text))
        for k in range(limit, 0, -1):
            if text.endswith(tag[:k]):
                best = max(best, k)
                break
    return best


def _clean_stream_text(text: str) -> str:
    for tag in JUNK_TAGS:
        text = text.replace(tag, "")
    text = text.lstrip()
    q = text.find("?")
    if 0 <= q < 8 and q + 1 < len(text) and text[q + 1].isupper():
        text = text[q + 1:]
    m = 0
    while m < len(text) and m < 4 and text[m].islower():
        m += 1
    if 0 < m <= 4 and m < len(text) and text[m].isupper():
        text = text[m:]
    i = text.find("Resposta:")
    if 0 <= i < START_BUFFER:
        text = text[i + len("Resposta:"):].lstrip(":").lstrip()
    return text


@torch.no_grad()
def generate_tokens(prompt_user: str, max_new_tokens: int, temperature: float, top_k: int | None, top_p: float | None, repetition_penalty: float = 1.2, context=None):
    prompt_text = build_prompt(prompt_user, context=context)
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    do_sample = not HAS_ADAPTER
    thread = Thread(target=model.generate, kwargs=dict(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=max(temperature, 1e-5),
        top_k=top_k or 0,
        top_p=top_p or 1.0,
        repetition_penalty=repetition_penalty if do_sample else 1.0,
        no_repeat_ngram_size=NO_REPEAT_NGRAM if do_sample else 0,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        stop_strings=list(STOP_STRINGS),
        tokenizer=tokenizer,
        streamer=streamer,
    ))
    thread.start()

    acc = ""
    emitted = 0
    for chunk in streamer:
        if not chunk:
            continue
        acc += chunk
        cleaned = _clean_stream_text(acc)
        stop_at = None
        for marker in CUT_MARKERS:
            i = cleaned.find(marker)
            if i != -1 and (stop_at is None or i < stop_at):
                stop_at = i
        if stop_at is not None:
            cleaned = cleaned[:stop_at]
        boundary = len(cleaned) - _partial_tag_len(cleaned)
        if boundary > emitted:
            if emitted == 0 and boundary < START_BUFFER and stop_at is None:
                continue
            yield cleaned[emitted:boundary]
            emitted = boundary
        if stop_at is not None:
            break

    thread.join()
    if emitted == 0:
        cleaned = _clean_stream_text(acc)
        for marker in CUT_MARKERS:
            i = cleaned.find(marker)
            if i != -1:
                cleaned = cleaned[:i]
                break
        cleaned = cleaned[:len(cleaned) - _partial_tag_len(cleaned)]
        if cleaned:
            yield cleaned


app = FastAPI(title=MODEL_LABEL)


class ChatRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 250
    temperature: float = 0.3
    top_k: int | None = 50
    top_p: float | None = 0.9
    repetition_penalty: float = 1.2


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

        context = None
        sources = []
        if retriever is not None and not _looks_generative(req.prompt):
            try:
                hits = await loop.run_in_executor(None, retriever.retrieve, req.prompt)
            except Exception as e:
                print(f"[rag] retrieve falhou: {e}")
                hits = []
            sources = [{"title": h["title"], "url": h["url"]} for h in hits]
            context = None if HAS_ADAPTER else (hits if hits else None)

        if sources:
            yield f"data: {json.dumps({'sources': sources})}\n\n"

        gen = generate_tokens(req.prompt, req.max_new_tokens, req.temperature, req.top_k, req.top_p, req.repetition_penalty, context=context)
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
