"""Avaliação do V8 pós-treino — métricas extrínsecas + amostras qualitativas.

Roda duas avaliações que o resumo original do TCC prometia e que agora vamos cumprir
pra apresentação (julho):

1. ROUGE-L e ROUGE-1/ROUGE-2 — compara sumários gerados vs. referência
2. BERTScore — similaridade semântica
3. Geração estruturada de amostras Q&A pra avaliação humana

Uso (a partir da raiz do projeto):
    ./webapp/.venv/bin/python gpt/eval_v8.py [--checkpoint checkpoint_gpt_v8.pth]
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken


# Raiz do projeto e arquitetura V8 (definida em model.py, mesmo diretório).
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import (
    GPTLanguageModel,
    block_size,
    device,
)


def load_model(checkpoint_path: Path):
    model = GPTLanguageModel().to(device)
    ck = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    if isinstance(ck, dict) and "model_state_dict" in ck:
        model.load_state_dict(ck["model_state_dict"])
        step = ck.get("step", "?")
        print(f"[eval] V8 carregado (step {step})")
    else:
        model.load_state_dict(ck)
        print(f"[eval] state_dict carregado")
    model.eval()
    return model


ENC = tiktoken.get_encoding("gpt2")
EOT_ID = ENC.eot_token


@torch.no_grad()
def generate(model, prompt: str, max_new: int = 150, temperature: float = 0.7,
             top_k: int = 50, stop_on_eot: bool = True) -> str:
    ids = ENC.encode(prompt) if prompt else [0]
    idx = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    for _ in range(max_new):
        cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
        logits, _ = model(cond)
        logits = logits[:, -1, :] / temperature
        if top_k:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("inf")
        probs = F.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, nxt), dim=1)
        if stop_on_eot and nxt.item() == EOT_ID:
            break
    out = ENC.decode(idx[0].tolist())
    return out.replace("<|endoftext|>", "").strip()


# ---------------------------------------------------------------
# Pares de referência pra ROUGE/BERTScore — adapte se quiser
# ---------------------------------------------------------------
SUMMARY_PAIRS = [
    {
        "source": "A Constituição Federal de 1988 estabelece que todos são iguais perante a lei, sem distinção de qualquer natureza, garantindo aos brasileiros e aos estrangeiros residentes no País a inviolabilidade do direito à vida, à liberdade, à igualdade, à segurança e à propriedade.",
        "reference": "A Constituição garante igualdade perante a lei e direitos fundamentais a brasileiros e estrangeiros residentes.",
    },
    {
        "source": "Machado de Assis é considerado um dos maiores escritores brasileiros. Dom Casmurro narra a história de Bentinho e Capitu, explorando o ciúme e a dúvida sobre a traição.",
        "reference": "Dom Casmurro de Machado de Assis conta a história de Bentinho e Capitu, abordando ciúme e dúvida.",
    },
    {
        "source": "Fotossíntese é o processo pelo qual plantas verdes, algas e algumas bactérias usam a energia da luz solar para transformar água e gás carbônico em glicose e oxigênio.",
        "reference": "A fotossíntese transforma luz solar, água e gás carbônico em glicose e oxigênio nas plantas.",
    },
    {
        "source": "O Brasil é o maior país da América do Sul, com área de cerca de 8,5 milhões de km². Possui grande diversidade cultural, étnica e biológica, abrigando a Floresta Amazônica.",
        "reference": "O Brasil é o maior país sul-americano, com grande diversidade cultural e abriga a Amazônia.",
    },
]


# Prompts pra geração Q&A usados na avaliação humana
QA_PROMPTS = [
    "Pergunta: O que é inteligência artificial?\n\nResposta:",
    "Pergunta: Quem foi Machado de Assis?\n\nResposta:",
    "Pergunta: Qual a capital do Brasil?\n\nResposta:",
    "Pergunta: O que é a Constituição Federal?\n\nResposta:",
    "Pergunta: Como fazer um bolo simples?\n\nResposta:",
    "Pergunta: Explique o que é fotossíntese em uma frase.\n\nResposta:",
    "Pergunta: Resuma a história de Dom Casmurro.\n\nResposta:",
    "Pergunta: O que é um modelo de linguagem?\n\nResposta:",
    "Pergunta: Liste três autores brasileiros famosos.\n\nResposta:",
    "Pergunta: Qual a função do coração no corpo humano?\n\nResposta:",
]


def compute_rouge(gens, refs):
    """ROUGE-1, ROUGE-2, ROUGE-L. Requer `rouge-score`."""
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        print("[eval] rouge-score não instalado. Pulando ROUGE.")
        return None
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    r1 = r2 = rl = 0.0
    for g, r in zip(gens, refs):
        s = scorer.score(r, g)
        r1 += s["rouge1"].fmeasure
        r2 += s["rouge2"].fmeasure
        rl += s["rougeL"].fmeasure
    n = len(gens)
    return {"rouge1": r1 / n, "rouge2": r2 / n, "rougeL": rl / n}


def compute_bertscore(gens, refs):
    """BERTScore F1 em PT-BR."""
    try:
        from bert_score import score as bscore
    except ImportError:
        print("[eval] bert-score não instalado. Pulando BERTScore.")
        return None
    # Usa o multilingual model pra pegar PT-BR sem baixar um PT específico
    P, R, F1 = bscore(gens, refs, model_type="bert-base-multilingual-cased", verbose=False)
    return {"bertscore_p": P.mean().item(), "bertscore_r": R.mean().item(), "bertscore_f1": F1.mean().item()}


def run_summary_eval(model):
    print("\n=== Avaliação de sumários (ROUGE + BERTScore) ===")
    gens = []
    refs = []
    for i, pair in enumerate(SUMMARY_PAIRS, start=1):
        prompt = f"Texto: {pair['source']}\n\nResumo:"
        gen = generate(model, prompt, max_new=80, temperature=0.3, top_k=20)
        # Extrai só o resumo, removendo o prompt
        gen_only = gen.split("Resumo:", 1)[-1].strip()
        gens.append(gen_only[:300])  # trunca
        refs.append(pair["reference"])
        print(f"\n[{i}] fonte:    {pair['source'][:100]}...")
        print(f"    gerado:   {gen_only[:200]}")
        print(f"    referência: {pair['reference']}")
    rouge = compute_rouge(gens, refs)
    if rouge:
        print(f"\n  ROUGE-1 F: {rouge['rouge1']:.3f}")
        print(f"  ROUGE-2 F: {rouge['rouge2']:.3f}")
        print(f"  ROUGE-L F: {rouge['rougeL']:.3f}")
    bert = compute_bertscore(gens, refs)
    if bert:
        print(f"  BERTScore F1: {bert['bertscore_f1']:.3f}")
    return {"rouge": rouge, "bertscore": bert, "gens": gens, "refs": refs}


def run_qa_samples(model):
    print("\n=== Amostras Q&A (pra avaliação humana) ===")
    out = []
    for i, prompt in enumerate(QA_PROMPTS, start=1):
        gen = generate(model, prompt, max_new=120, temperature=0.7, top_k=50)
        resposta = gen.split("Resposta:", 1)[-1].strip()
        out.append({"id": i, "prompt": prompt, "resposta": resposta})
        print(f"\n[{i}] {prompt.split(chr(10))[0]}")
        print(f"    → {resposta[:300]}")
    return out


def make_human_eval_form(samples, output_path: Path):
    """Gera um CSV pra avaliadores humanos preencherem."""
    import csv
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "pergunta", "resposta", "coerencia_1a5", "fluencia_1a5", "utilidade_1a5", "comentarios"])
        for s in samples:
            perg = s["prompt"].replace("Pergunta: ", "").replace("\n\nResposta:", "")
            w.writerow([s["id"], perg, s["resposta"], "", "", "", ""])
    print(f"\n[eval] formulário de avaliação humana salvo em {output_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(ROOT / "checkpoint_gpt_v8.pth"))
    ap.add_argument("--out", default=str(ROOT / "outputs" / "eval_v8.json"))
    ap.add_argument("--form", default=str(ROOT / "outputs" / "avaliacao_humana_v8.csv"))
    args = ap.parse_args()

    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        print(f"[eval] checkpoint não encontrado: {ckpt}")
        sys.exit(1)

    model = load_model(ckpt)

    resultado = {
        "checkpoint": str(ckpt),
        "summary_eval": run_summary_eval(model),
        "qa_samples": run_qa_samples(model),
    }

    # Salva resultados
    out_path = Path(args.out)
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        # BERTScore/ROUGE retornam floats; gens/refs são strings — JSON OK
        json.dump(resultado, f, ensure_ascii=False, indent=2)
    print(f"\n[eval] resultados salvos em {out_path}")

    # Gera formulário pra avaliação humana
    make_human_eval_form(resultado["qa_samples"], Path(args.form))


if __name__ == "__main__":
    main()
