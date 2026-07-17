"""Monta data/qa_demo_ptbr.txt: pares Q&A curados dos assuntos da demo.

Para cada assunto: uma resposta curta e factualmente correta + várias
formulações da pergunta (ajuda o modelo a responder mesmo se a banca perguntar
diferente). Saída no mesmo formato do qa_brasil_ptbr.txt.
"""
from pathlib import Path

OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "qa_demo_ptbr.txt"

# (resposta, [formulações da pergunta])
TOPICS = [
    (
        "Machado de Assis (1839–1908) foi um escritor brasileiro, autor de romances, "
        "contos, poesias e peças de teatro, considerado o maior nome da literatura brasileira. "
        "Nasceu e morreu no Rio de Janeiro e foi o fundador e primeiro presidente da Academia "
        "Brasileira de Letras. Suas principais obras incluem Memórias Póstumas de Brás Cubas, "
        "Dom Casmurro, Quincas Borba e O Alienista.",
        [
            "Quem foi Machado de Assis?",
            "Quem é Machado de Assis?",
            "Fale sobre Machado de Assis.",
            "Quem foi o maior escritor da literatura brasileira?",
            "O que você sabe sobre Machado de Assis?",
        ],
    ),
    (
        "Dom Casmurro é um romance de Machado de Assis publicado em 1899, considerado uma "
        "das maiores obras da literatura brasileira. É narrado por Bento Santiago, o Bentinho, "
        "e conta a história de seu casamento com Capitu, girando em torno da dúvida sobre uma "
        "suposta traição. É um clássico do Realismo brasileiro.",
        [
            "Quem escreveu Dom Casmurro?",
            "Quem é o autor de Dom Casmurro?",
            "Quem escreveu o romance Dom Casmurro?",
            "Dom Casmurro é uma obra de qual escritor?",
            "Fale sobre o livro Dom Casmurro.",
        ],
    ),
    (
        "Memórias Póstumas de Brás Cubas é um romance de Machado de Assis publicado em 1881, "
        "considerado o marco inicial do Realismo brasileiro. É narrado pelo próprio Brás Cubas, "
        "um 'defunto autor', em tom irônico e não-linear. É apontado como uma das obras-primas de Machado.",
        [
            "Quem escreveu Memórias Póstumas de Brás Cubas?",
            "O que é Memórias Póstumas de Brás Cubas?",
            "Quem é o autor de Memórias Póstumas de Brás Cubas?",
            "Fale sobre o romance Memórias Póstumas de Brás Cubas.",
        ],
    ),
    (
        "A Academia Brasileira de Letras (ABL) foi fundada em 1897 e tem como primeiro "
        "presidente Machado de Assis. Reúne quarenta membros chamados imortais e tem por objetivo "
        "cultivar a língua e a literatura nacional.",
        [
            "Quem fundou a Academia Brasileira de Letras?",
            "Quem foi o primeiro presidente da Academia Brasileira de Letras?",
            "Machado de Assis fundou qual instituição?",
            "O que é a Academia Brasileira de Letras?",
        ],
    ),
    (
        "A inteligência artificial (IA) é a área da ciência da computação que desenvolve "
        "sistemas capazes de realizar tarefas que normalmente exigem inteligência humana, como "
        "raciocinar, aprender, perceber e compreender linguagem. Engloba o aprendizado de máquina, "
        "o aprendizado profundo, o processamento de linguagem natural e a visão computacional.",
        [
            "O que é inteligência artificial?",
            "Defina inteligência artificial.",
            "O que significa IA?",
            "Explique o que é a inteligência artificial.",
            "O que é inteligência artificial e onde é aplicada?",
        ],
    ),
    (
        "O aprendizado de máquina é uma área da inteligência artificial em que os sistemas "
        "aprendem padrões a partir de dados, em vez de serem programados explicitamente para cada "
        "tarefa. Compreende o aprendizado supervisionado, o não supervisionado e o por reforço, "
        "e é a base da maioria dos sistemas modernos de IA.",
        [
            "O que é aprendizado de máquina?",
            "Defina aprendizado de máquina.",
            "O que é machine learning?",
            "Explique o que é o aprendizado de máquina.",
        ],
    ),
    (
        "A fotossíntese é o processo pelo qual plantas, algas e algumas bactérias convertem "
        "a energia da luz solar em energia química. Usando dióxido de carbono e água, produzem "
        "glicose e liberam oxigênio. Ocorre nos cloroplastos, graças à clorofila, e é essencial "
        "para a vida na Terra.",
        [
            "O que é fotossíntese?",
            "Como funciona a fotossíntese?",
            "Explique o processo de fotossíntese.",
            "Defina fotossíntese.",
            "O que as plantas fazem na fotossíntese?",
        ],
    ),
    (
        "A capital do Brasil é Brasília. Inaugurada em 1960, é uma cidade planejada na região "
        "Centro-Oeste, projetada pelo urbanista Lúcio Costa e pelo arquiteto Oscar Niemeyer. "
        "Brasília substituiu o Rio de Janeiro como capital do país.",
        [
            "Qual é a capital do Brasil?",
            "Qual a capital do Brasil?",
            "Qual é a capital brasileira?",
            "Desde quando Brasília é a capital do Brasil?",
            "Quem projetou Brasília?",
        ],
    ),
    (
        "O Realismo foi um movimento literário brasileiro da segunda metade do século XIX, "
        "caracterizado pela objetividade e pela crítica social. Teve como precursor o romance "
        "O Mulato, de Aluísio Azevedo, em 1881, e como principal representante Machado de Assis, "
        "autor de Memórias Póstumas de Brás Cubas e Dom Casmurro.",
        [
            "O que foi o Realismo no Brasil?",
            "Quais os principais autores do Realismo brasileiro?",
            "Fale sobre o Realismo na literatura brasileira.",
            "Quem foram os principais escritores realistas brasileiros?",
        ],
    ),
]


def main():
    out = []
    n_examples = 0
    for answer, questions in TOPICS:
        for q in questions:
            out.append(f"Pergunta: {q.strip()}\n\nResposta: {answer.strip()}\n<|endoftext|>")
            n_examples += 1
    OUT_PATH.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"[demo] {n_examples} exemplos, {len(TOPICS)} assuntos -> {OUT_PATH}")


if __name__ == "__main__":
    main()
