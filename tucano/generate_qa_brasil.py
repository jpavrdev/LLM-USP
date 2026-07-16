"""Gera dataset Q&A curado sobre Brasil (cultura, literatura, geografia, história).

Estratégia: ~50 fatos essenciais com 4-5 formulações de pergunta cada, produzindo
~200 pares. Múltiplas formulações ajudam o modelo a generalizar o fato além da
sintaxe exata da pergunta.

Formato idêntico ao qa_dolly_ptbr.txt / qa_alpaca_ptbr.txt:
    Pergunta: <q>

    Resposta: <r>
    <|endoftext|>
"""
from pathlib import Path


OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "qa_brasil_ptbr.txt"


# ===============================================================
# Fatos: lista de (resposta_completa, [variações_da_pergunta])
# ===============================================================
FACTS = [
    # --------- Literatura ---------
    ("Dom Casmurro é um romance de Machado de Assis, publicado em 1899. É considerado uma das maiores obras da literatura brasileira e narra a história de Bentinho e Capitu, explorando o ciúme e a dúvida sobre uma possível traição.", [
        "Quem é o autor de Dom Casmurro?",
        "Quem escreveu Dom Casmurro?",
        "Qual autor escreveu a obra Dom Casmurro?",
        "Dom Casmurro é uma obra de qual escritor?",
        "Quem é o autor do romance Dom Casmurro?",
    ]),
    ("Memórias Póstumas de Brás Cubas, publicado em 1881, é um romance de Machado de Assis narrado por um defunto-autor. É marco do Realismo brasileiro.", [
        "Quem escreveu Memórias Póstumas de Brás Cubas?",
        "Quem é o autor de Memórias Póstumas de Brás Cubas?",
        "De qual escritor é o livro Memórias Póstumas de Brás Cubas?",
        "Qual autor escreveu Memórias Póstumas?",
    ]),
    ("Machado de Assis (1839-1908) foi um escritor brasileiro, fundador da Academia Brasileira de Letras e considerado o maior nome da literatura brasileira. Entre suas obras estão Dom Casmurro, Memórias Póstumas de Brás Cubas e Quincas Borba.", [
        "Quem foi Machado de Assis?",
        "Quem é Machado de Assis?",
        "Fale sobre Machado de Assis.",
        "O que você sabe sobre Machado de Assis?",
    ]),
    ("Clarice Lispector (1920-1977) foi uma escritora brasileira nascida na Ucrânia, conhecida pelo intimismo e pela introspecção. Escreveu A Hora da Estrela, Perto do Coração Selvagem e A Paixão Segundo G.H.", [
        "Quem foi Clarice Lispector?",
        "Quem é Clarice Lispector?",
        "Fale sobre Clarice Lispector.",
        "O que você sabe sobre Clarice Lispector?",
    ]),
    ("A Hora da Estrela é um romance de Clarice Lispector, publicado em 1977. Conta a história de Macabéa, uma jovem nordestina migrante que vive no Rio de Janeiro.", [
        "Quem escreveu A Hora da Estrela?",
        "Quem é o autor de A Hora da Estrela?",
        "De qual autora é o livro A Hora da Estrela?",
    ]),
    ("Jorge Amado (1912-2001) foi um escritor brasileiro baiano, autor de Capitães da Areia, Gabriela Cravo e Canela, Dona Flor e Seus Dois Maridos e Tieta do Agreste. É um dos escritores brasileiros mais traduzidos no exterior.", [
        "Quem foi Jorge Amado?",
        "Quem é Jorge Amado?",
        "Fale sobre Jorge Amado.",
    ]),
    ("Capitães da Areia é um romance de Jorge Amado, publicado em 1937, sobre um grupo de meninos de rua em Salvador.", [
        "Quem escreveu Capitães da Areia?",
        "Quem é o autor de Capitães da Areia?",
    ]),
    ("Gabriela, Cravo e Canela é um romance de Jorge Amado, publicado em 1958, ambientado em Ilhéus, Bahia.", [
        "Quem escreveu Gabriela, Cravo e Canela?",
        "Quem é o autor de Gabriela, Cravo e Canela?",
    ]),
    ("Guimarães Rosa (1908-1967) foi um escritor brasileiro mineiro, autor de Grande Sertão: Veredas. Sua obra é marcada por inovação linguística e ambientação sertaneja.", [
        "Quem foi Guimarães Rosa?",
        "Quem é Guimarães Rosa?",
        "Fale sobre João Guimarães Rosa.",
    ]),
    ("Grande Sertão: Veredas é o maior romance de Guimarães Rosa, publicado em 1956, sobre o jagunço Riobaldo narrando sua vida no sertão mineiro.", [
        "Quem escreveu Grande Sertão: Veredas?",
        "Quem é o autor de Grande Sertão: Veredas?",
    ]),
    ("Graciliano Ramos (1892-1953) foi um escritor alagoano, autor de Vidas Secas, São Bernardo e Memórias do Cárcere. É um dos principais nomes do Regionalismo brasileiro.", [
        "Quem foi Graciliano Ramos?",
        "Quem escreveu Vidas Secas?",
    ]),
    ("Carlos Drummond de Andrade (1902-1987) foi um dos maiores poetas brasileiros. Autor de poemas como No Meio do Caminho e de livros como Sentimento do Mundo e A Rosa do Povo.", [
        "Quem foi Carlos Drummond de Andrade?",
        "Quem é Carlos Drummond de Andrade?",
        "Fale sobre Drummond.",
    ]),
    ("Manuel Bandeira (1886-1968) foi um poeta brasileiro do modernismo, autor de Libertinagem e A Cinza das Horas. Participou da Semana de Arte Moderna de 1922.", [
        "Quem foi Manuel Bandeira?",
        "Fale sobre Manuel Bandeira.",
    ]),
    ("José de Alencar (1829-1877) foi um escritor romântico brasileiro, autor de Iracema, O Guarani e Senhora. É considerado o principal nome do Romantismo no Brasil.", [
        "Quem foi José de Alencar?",
        "Quem escreveu Iracema?",
        "Quem é o autor de O Guarani?",
    ]),
    ("A Semana de Arte Moderna de 1922 foi um evento realizado em São Paulo que marcou o início do Modernismo no Brasil. Participaram Mário de Andrade, Oswald de Andrade, Anita Malfatti, Tarsila do Amaral e Heitor Villa-Lobos, entre outros.", [
        "O que foi a Semana de Arte Moderna de 1922?",
        "O que foi a Semana de 22?",
        "Fale sobre a Semana de Arte Moderna.",
    ]),

    # --------- Geografia ---------
    ("Brasília é a capital federal do Brasil, localizada no Distrito Federal, na região Centro-Oeste. Foi inaugurada em 21 de abril de 1960 durante o governo de Juscelino Kubitschek.", [
        "Qual é a capital do Brasil?",
        "Qual a capital do Brasil?",
        "Onde fica a capital do Brasil?",
        "Qual a capital federal do Brasil?",
    ]),
    ("São Paulo é a maior cidade do Brasil e da América do Sul, com mais de 12 milhões de habitantes. É o principal centro econômico e financeiro do país.", [
        "Qual é a maior cidade do Brasil?",
        "Qual a cidade mais populosa do Brasil?",
        "Qual a maior cidade brasileira?",
    ]),
    ("O Brasil possui 26 estados e o Distrito Federal, totalizando 27 unidades federativas. Está dividido em cinco regiões: Norte, Nordeste, Centro-Oeste, Sudeste e Sul.", [
        "Quantos estados tem o Brasil?",
        "Quantas unidades federativas o Brasil possui?",
        "Quantas regiões tem o Brasil?",
    ]),
    ("O Rio Amazonas é o maior rio do mundo em volume de água e o mais extenso do Brasil. Nasce nos Andes peruanos e deságua no Oceano Atlântico.", [
        "Qual é o maior rio do Brasil?",
        "Qual é o rio mais extenso do Brasil?",
        "Qual o maior rio do mundo?",
    ]),
    ("A Floresta Amazônica é a maior floresta tropical do mundo, cobrindo cerca de 60% do território brasileiro. Abriga imensa biodiversidade e desempenha papel crítico no equilíbrio climático global.", [
        "O que é a Floresta Amazônica?",
        "O que é a Amazônia?",
        "Qual a importância da Floresta Amazônica?",
    ]),
    ("Os seis biomas brasileiros são: Amazônia, Cerrado, Caatinga, Mata Atlântica, Pampa e Pantanal.", [
        "Quais são os biomas brasileiros?",
        "Quais os principais biomas do Brasil?",
        "Liste os biomas do Brasil.",
    ]),
    ("O Pantanal é a maior planície alagável do mundo, localizado entre Mato Grosso e Mato Grosso do Sul, além de partes da Bolívia e do Paraguai.", [
        "O que é o Pantanal?",
        "Onde fica o Pantanal?",
    ]),
    ("O Cerrado é o segundo maior bioma do Brasil, ocupando principalmente a região Centro-Oeste. Considerado uma savana tropical, abriga mais de 11 mil espécies de plantas.", [
        "O que é o Cerrado?",
        "Onde fica o Cerrado brasileiro?",
    ]),
    ("O Rio São Francisco, conhecido como Velho Chico, atravessa Minas Gerais, Bahia, Pernambuco, Alagoas e Sergipe, deságua no Atlântico entre AL e SE.", [
        "Qual é o Rio São Francisco?",
        "O que é o Rio São Francisco?",
        "Onde fica o Velho Chico?",
    ]),
    ("O Pico da Neblina, localizado no Amazonas na fronteira com a Venezuela, é o ponto mais alto do Brasil com 2.995,30 metros.", [
        "Qual é o ponto mais alto do Brasil?",
        "Qual a montanha mais alta do Brasil?",
    ]),
    ("As capitais brasileiras das regiões Sul são: Porto Alegre (Rio Grande do Sul), Florianópolis (Santa Catarina) e Curitiba (Paraná).", [
        "Quais são as capitais dos estados do Sul do Brasil?",
        "Cite as capitais da região Sul do Brasil.",
    ]),
    ("As capitais brasileiras do Nordeste são: Salvador (BA), Aracaju (SE), Maceió (AL), Recife (PE), João Pessoa (PB), Natal (RN), Fortaleza (CE), Teresina (PI) e São Luís (MA).", [
        "Quais são as capitais dos estados do Nordeste?",
        "Cite as capitais da região Nordeste.",
    ]),
    ("O litoral brasileiro possui cerca de 7.491 km de extensão, banhado pelo Oceano Atlântico.", [
        "Qual a extensão do litoral brasileiro?",
        "Quantos quilômetros tem o litoral do Brasil?",
    ]),

    # --------- História ---------
    ("O Brasil foi descoberto pelos portugueses em 22 de abril de 1500, quando a frota de Pedro Álvares Cabral chegou ao litoral da atual Bahia.", [
        "Quando o Brasil foi descoberto?",
        "Quem descobriu o Brasil?",
        "Em que ano o Brasil foi descoberto?",
    ]),
    ("A Independência do Brasil foi proclamada por Dom Pedro I em 7 de setembro de 1822, às margens do Rio Ipiranga, em São Paulo.", [
        "Quando foi a Independência do Brasil?",
        "Em que ano o Brasil se tornou independente?",
        "Quem proclamou a Independência do Brasil?",
    ]),
    ("A Proclamação da República aconteceu em 15 de novembro de 1889, pelo Marechal Deodoro da Fonseca, encerrando o Império e o reinado de Dom Pedro II.", [
        "Quando foi a Proclamação da República?",
        "Quem proclamou a República no Brasil?",
    ]),
    ("A Lei Áurea, assinada pela Princesa Isabel em 13 de maio de 1888, aboliu a escravidão no Brasil.", [
        "O que foi a Lei Áurea?",
        "Quando a escravidão foi abolida no Brasil?",
        "Quem assinou a Lei Áurea?",
    ]),
    ("A Ditadura Militar no Brasil durou de 1964 a 1985, iniciada pelo golpe contra o presidente João Goulart. Foi marcada pelo AI-5, perseguições políticas, censura e violações de direitos humanos.", [
        "O que foi a Ditadura Militar no Brasil?",
        "Quando foi a Ditadura Militar brasileira?",
        "Fale sobre o período da Ditadura Militar no Brasil.",
    ]),
    ("A Era Vargas (1930-1945) foi o período em que Getúlio Vargas governou o Brasil. Incluiu o Estado Novo (1937-1945) e trouxe direitos trabalhistas consolidados na CLT.", [
        "O que foi a Era Vargas?",
        "Quem foi Getúlio Vargas?",
    ]),
    ("A Constituição Federal de 1988, apelidada de Constituição Cidadã, é a atual constituição brasileira. Foi promulgada em 5 de outubro de 1988, encerrando o regime militar e consolidando direitos fundamentais.", [
        "O que é a Constituição de 1988?",
        "Quando foi promulgada a Constituição Cidadã?",
        "Fale sobre a Constituição Federal do Brasil.",
    ]),
    ("Juscelino Kubitschek (1902-1976), o JK, foi presidente do Brasil entre 1956 e 1961. Ficou conhecido pelo Plano de Metas e pela construção de Brasília, que se tornou capital em 1960.", [
        "Quem foi Juscelino Kubitschek?",
        "O que JK fez como presidente?",
        "Quem construiu Brasília?",
    ]),
    ("A Semana de Arte Moderna de 1922 aconteceu em fevereiro, no Teatro Municipal de São Paulo, marcando o início do Modernismo brasileiro.", [
        "Em que ano aconteceu a Semana de Arte Moderna?",
        "Onde foi realizada a Semana de Arte Moderna?",
    ]),
    ("As Diretas Já (1983-1984) foi um movimento popular que exigiu eleições diretas para presidente durante o fim da Ditadura Militar. A emenda Dante de Oliveira foi rejeitada, mas o movimento acelerou a redemocratização.", [
        "O que foi o movimento Diretas Já?",
        "Quando aconteceu o movimento Diretas Já?",
    ]),

    # --------- Cultura e música ---------
    ("A Bossa Nova é um gênero musical brasileiro surgido no fim dos anos 1950, com Tom Jobim, João Gilberto e Vinicius de Moraes entre seus principais nomes. A canção Garota de Ipanema é um ícone do gênero.", [
        "O que é a Bossa Nova?",
        "Quem são os principais artistas da Bossa Nova?",
        "Fale sobre a Bossa Nova.",
    ]),
    ("Tom Jobim (1927-1994), Antônio Carlos Jobim, foi um dos maiores compositores brasileiros, coautor de Garota de Ipanema, Águas de Março e Chega de Saudade.", [
        "Quem foi Tom Jobim?",
        "Quem é Antônio Carlos Jobim?",
    ]),
    ("O Tropicalismo foi um movimento cultural brasileiro do final dos anos 1960, liderado por Caetano Veloso, Gilberto Gil, Gal Costa, Tom Zé e Os Mutantes. Misturava música popular com experimentação.", [
        "O que foi o Tropicalismo?",
        "Quem foram os líderes do Tropicalismo?",
    ]),
    ("O Carnaval brasileiro é uma das maiores festas populares do mundo. Em destaque estão os desfiles de escolas de samba do Rio de Janeiro e de São Paulo, o Carnaval de rua e o Carnaval de Salvador com seus trios elétricos.", [
        "O que é o Carnaval brasileiro?",
        "Como é o Carnaval no Brasil?",
    ]),
    ("O Samba é um gênero musical brasileiro com raízes africanas, consolidado no Rio de Janeiro no início do século XX. Ícone do Carnaval e da identidade cultural nacional.", [
        "O que é o Samba?",
        "Qual a origem do Samba?",
    ]),
    ("Pelé (Edson Arantes do Nascimento, 1940-2022) foi um jogador de futebol brasileiro, considerado um dos maiores da história. Tricampeão mundial (1958, 1962 e 1970) e ídolo do Santos FC.", [
        "Quem foi Pelé?",
        "Quem é o Rei do Futebol?",
        "Fale sobre Pelé.",
    ]),
    ("A seleção brasileira de futebol é pentacampeã mundial, com títulos em 1958, 1962, 1970, 1994 e 2002.", [
        "Quantos títulos mundiais tem a seleção brasileira?",
        "Em que anos o Brasil foi campeão mundial de futebol?",
    ]),

    # --------- Instituições, ciência e economia ---------
    ("A Universidade de São Paulo (USP), fundada em 1934, é a maior e mais importante universidade do Brasil, referência em pesquisa e ensino na América Latina.", [
        "O que é a USP?",
        "Quando foi fundada a USP?",
    ]),
    ("A Petrobras (Petróleo Brasileiro S.A.), fundada em 1953 no governo de Getúlio Vargas, é uma estatal brasileira de energia e petróleo.", [
        "O que é a Petrobras?",
        "Quando foi criada a Petrobras?",
    ]),
    ("A Embraer é uma empresa brasileira fabricante de aviões, fundada em 1969 em São José dos Campos, São Paulo. É uma das maiores do mundo no setor.", [
        "O que é a Embraer?",
        "O que faz a Embraer?",
    ]),
    ("O Sistema Único de Saúde (SUS), criado pela Constituição de 1988, é a rede pública de saúde brasileira, que oferece atendimento universal e gratuito a toda a população.", [
        "O que é o SUS?",
        "Fale sobre o Sistema Único de Saúde.",
        "Quando foi criado o SUS?",
    ]),
    ("A Lei Geral de Proteção de Dados (LGPD), Lei nº 13.709/2018, regulamenta o tratamento de dados pessoais no Brasil, inspirada no GDPR europeu.", [
        "O que é a LGPD?",
        "O que a LGPD estabelece?",
        "Quando a LGPD entrou em vigor?",
    ]),
    ("Oswaldo Cruz (1872-1917) foi um médico e sanitarista brasileiro, responsável pelo combate à febre amarela e à peste bubônica no Rio de Janeiro no início do século XX. Fundou o Instituto Oswaldo Cruz.", [
        "Quem foi Oswaldo Cruz?",
    ]),
    ("Carlos Chagas (1879-1934) foi um médico brasileiro que descobriu a doença de Chagas em 1909, causada pelo Trypanosoma cruzi.", [
        "Quem foi Carlos Chagas?",
        "Quem descobriu a doença de Chagas?",
    ]),
    ("O Real é a moeda oficial do Brasil desde 1994, quando o Plano Real foi implementado no governo de Itamar Franco, tendo Fernando Henrique Cardoso como ministro da Fazenda.", [
        "Qual é a moeda do Brasil?",
        "Quando o Real foi criado?",
        "O que foi o Plano Real?",
    ]),

    # --------- Presidentes modernos ---------
    ("Fernando Henrique Cardoso (FHC), sociólogo e ex-ministro da Fazenda, foi presidente do Brasil entre 1995 e 2002, responsável pela estabilização da economia via Plano Real.", [
        "Quem foi Fernando Henrique Cardoso?",
        "Quem criou o Plano Real?",
    ]),
    ("Luiz Inácio Lula da Silva, ex-sindicalista, foi presidente do Brasil entre 2003 e 2010 e voltou a ser eleito em 2022, para mandato iniciado em 2023.", [
        "Quem é Lula?",
        "Quem é Luiz Inácio Lula da Silva?",
    ]),
    ("Dilma Rousseff foi presidente do Brasil entre 2011 e 2016, primeira mulher a ocupar o cargo. Sofreu impeachment em 2016.", [
        "Quem foi Dilma Rousseff?",
        "Qual foi a primeira mulher presidente do Brasil?",
    ]),

    # --------- Gastronomia ---------
    ("A feijoada é um prato típico brasileiro, feito com feijão preto e cortes de carne de porco (costela, linguiça, pé, orelha). Acompanha arroz, couve, farofa e laranja.", [
        "O que é feijoada?",
        "Qual o prato típico do Brasil?",
        "Como é feita a feijoada?",
    ]),
    ("O brigadeiro é um doce brasileiro feito com leite condensado, chocolate em pó e manteiga, moldado em bolinhas e coberto com granulado. Criado na década de 1940.", [
        "O que é brigadeiro?",
        "Como se faz brigadeiro?",
    ]),
    ("O açaí é um fruto amazônico muito consumido no Brasil, especialmente em forma de polpa congelada servida com granola, banana e mel.", [
        "O que é açaí?",
        "Qual é a origem do açaí?",
    ]),
    ("A moqueca é um prato típico brasileiro, em versões baiana (com azeite de dendê e leite de coco) e capixaba (sem dendê). Leva peixe ou frutos do mar com tomate, cebola, pimentão e coentro.", [
        "O que é moqueca?",
        "Qual a diferença entre moqueca baiana e capixaba?",
    ]),

    # --------- Geográficas adicionais ---------
    ("O Rio de Janeiro foi a capital do Brasil entre 1763 e 1960, quando o título passou para Brasília.", [
        "Qual foi a capital do Brasil antes de Brasília?",
        "Quando o Rio de Janeiro deixou de ser a capital?",
    ]),
    ("Minas Gerais é um estado da região Sudeste cuja capital é Belo Horizonte, conhecido pelo patrimônio histórico de Ouro Preto e pela culinária (pão de queijo, feijão tropeiro).", [
        "Qual a capital de Minas Gerais?",
        "Fale sobre Minas Gerais.",
    ]),
    ("Salvador foi a primeira capital do Brasil, de 1549 a 1763. É hoje a capital da Bahia, famosa pelo Pelourinho e pelo Carnaval.", [
        "Qual foi a primeira capital do Brasil?",
        "Qual a capital da Bahia?",
    ]),
    ("Os quatro estados da região Sudeste são: São Paulo, Rio de Janeiro, Minas Gerais e Espírito Santo.", [
        "Quais estados compõem a região Sudeste?",
        "Cite os estados do Sudeste brasileiro.",
    ]),
]


def format_example(q: str, r: str) -> str:
    return f"Pergunta: {q}\n\nResposta: {r}\n<|endoftext|>\n"


def main():
    total = 0
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for answer, questions in FACTS:
            for q in questions:
                f.write(format_example(q, answer))
                total += 1
    size_mb = OUT_PATH.stat().st_size / 1e6
    print(f"[qa-brasil] {len(FACTS)} fatos × ~{total/len(FACTS):.1f} variações = {total} pares")
    print(f"[qa-brasil] salvo em {OUT_PATH} ({size_mb:.3f} MB)")


if __name__ == "__main__":
    main()
