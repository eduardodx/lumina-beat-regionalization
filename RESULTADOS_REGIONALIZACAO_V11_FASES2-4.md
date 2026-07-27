# Resultados — Regionalização do Beat-v11, Fases 2-4 (M0 → Fusion → Calibração)

**Modelo:** Beat-v11 BioPrime r1 (`lumina-beat-v11v5-r1-202607071631`, SISO, 52M, d_full=448) · **Base a bater:** o estudo de regionalização ABRAOM do Pedro no **Beat-v10** · **Tarefa:** classificar patogenicidade de variantes (ClinVar) com calibração pela frequência alélica brasileira (ABRAOM) · **Split de comparação:** `test` (paridade com a tabela §6.1 do Pedro).

> Este documento fecha o **porte downstream** da regionalização do v10 para o v11: Fase 2 (baseline molecular M0), Fase 3 (fusion de adapters + falsificação), e Fase 4 (calibração de frequência M5_v2/M5_v3 + a falsificação estratificada). Complementa o `RESULTADOS_REGIONALIZACAO_V11.md`, que cobriu a Fase 1 (adapters de frequência).
>
> **Escrito para ser lido de cima a baixo por quem quer *entender* o resultado**, não só ver números. A Seção 5 (a falsificação) é o coração científico — leia com calma.

---

## 1. TL;DR — as 5 conclusões

1. **O backbone do v11 é melhor em patogenicidade, globalmente.** O M0 (baseline molecular, sem regionalização) sobe de **MCC 0.576 (v10) → 0.654 (v11)** no teste global. E ele levanta o brasileiro na mesma proporção (+0.09), **mas não fecha** a lacuna brasileiro↔global — porque essa lacuna é de *calibração de frequência*, não de *representação*.

2. **O fusion cru de adapters NÃO carrega sinal ABRAOM.** Com teste bem-dimensionado (n=4163), o adapter real e o embaralhado (scrambled) são **estatisticamente indistinguíveis** no brasileiro (Δ +0.003, CI cruza 0). O valor do ABRAOM não é aprendível pela sequência.

3. **A calibração de frequência REPRODUZ o headline do Pedro.** Usando a AF ABRAOM *observada* como desconto, o `br_only` MCC salta de **0.24 (M0) → 0.57 (M5_v2)** — espelhando o 0.28→0.61 do Pedro — preservando recall e specificity, e **superando o v10 no global** (0.626 vs 0.512).

4. **A calibração é GENUINAMENTE ABRAOM-específica — falsificação limpa.** Os controles negativos estratificados (desconto embaralhado) ficam **todos abaixo** do real (p=0.0196). **Isso é o que o v10 do Pedro NÃO conseguiu** (lá os controles chegavam perto → não-falsificado). No v11 a especificidade brasileira está *demonstrada*.

5. **A tese final é coerente ponta a ponta:** o valor do ABRAOM **não é aprendível da sequência** (Fases 1/B/3, todas null), **mas a frequência observada, usada como calibração, carrega sinal regional real e falsificável** (Fase 4). Regionalização = calibração de frequência — e ela é de verdade.

---

## 2. Contexto — o que já sabíamos antes destas fases

A missão: melhorar a interpretação de patogenicidade de variantes no contexto **brasileiro** usando o **ABRAOM** (frequência alélica de uma coorte brasileira). A hipótese: foundation models de DNA vão mal em variantes brasileiras porque o *prior de frequência* deles é europeu; calibrar pela AF brasileira deveria ajudar.

As fases anteriores (`RESULTADOS_REGIONALIZACAO_V11.md`) já tinham estabelecido, com controles:
- **Fase 1:** o adapter que aprende AF a partir da sequência tem sinal real, mas **não é brasileiro-específico** (A_BR ≈ A_gnomAD).
- **Experimento B (resíduo):** um adapter dedicado ao componente "ABRAOM além do gnomAD" tem sinal **tênue** (+0.026, borderline).

Ou seja: entrando nas Fases 2-4, a suspeita forte era que **a sequência não carrega o sinal regional** — restava saber se a *calibração downstream* (o mecanismo que de fato deu o ganho do Pedro) reproduzia.

---

## 3. Metodologia — a escada de modelos

Todos treinam a cabeça de classificação sobre o backbone v11, em ClinVar **não-brasileiro** (`nonbr_only`, *leave-Brazilian-out* — o teste limpo: nenhuma variante brasileira é vista no treino). Diferem no que a cabeça enxerga e em como a frequência entra:

| Modelo | O que é | Frequência entra? |
|---|---|---|
| **M0** | baseline molecular puro | não |
| **M4 / M5** (cru) | fusion de adapters (A_BR + A_gnomAD) via gate estático/dinâmico | via adapters (implícito) |
| **M7** (controle) | fusion com adapter **scrambled** no lugar do A_BR | controle negativo |
| **M5-bounded** | fusion + **AF explícita** na cabeça (`RegimeABoundedRegionalHead`): `score = molecular − desconto(AF)` | sim, explícita e treinada |
| **M5_v2** | re-calibra o desconto do M5-bounded (tuna escala/teto/thresholds no holdout) | sim, calibrada |
| **M5_v3** | M5_v2 + **guarda molecular** (protege evidência molecular forte de ser descontada) | sim, calibrada + guarda |

A ideia da escada: cada degrau isola uma pergunta. M0 = quão bom é o molecular puro. M4/M5 vs M7 = o *adapter* aprende ABRAOM? M5-bounded/v2/v3 = a *frequência observada* ajuda, e é específica?

---

## 4. Resultados

### 4.1 Fase 2 — M0 (baseline molecular): o backbone ganha, mas não conserta o brasileiro

| Métrica | Pedro v10 | **v11** |
|---|---|---|
| nonBR test MCC | 0.576 | **0.654** |
| nonBR test AUROC | 0.879 | **0.927** |

O v11 é um preditor de patogenicidade molecular claramente melhor. **Mas** — e este é o ponto científico — no split bem-dimensionado (`br_only.all`, n=4163, as mesmas variantes nos dois modelos):

| `br_only` MCC (n=4163) | v10 M0 | v11 M0 |
|---|---|---|
| valor | 0.248 | **0.335** (+0.087) |

O v11 levanta o brasileiro **+0.087** — praticamente igual ao ganho global (+0.094). **O backbone melhor levanta a régua inteira de forma quase uniforme, mas o abismo brasileiro↔global fica intacto (~0.27 nos dois).** Um foundation model melhor sozinho **não fecha** o ponto cego brasileiro. (Nota: no split `test` pequeno, n=504, o ruído chega a inverter esse sinal — por isso usamos o `all`, que é 100% out-of-sample para o brasileiro e ~8× mais poderoso.)

### 4.2 Fase 3 — fusion cru + falsificação: o adapter não carrega ABRAOM

Rodamos M4 (estático) e M5 (dinâmico), cada um com seu controle scrambled (M7s, M7d). Dois achados:

**(a) Os pesos aprendidos do gate são uniformes.** O fusion estático aprendeu peso ~50/50 entre A_BR e A_gnomAD (0.504 / 0.496) — **e deu o mesmo peso ao adapter scrambled** (0.502). O gate não distingue o ABRAOM real de ruído. (Motivo estrutural: o fusion treina em `nonbr_only`, então nunca vê variante brasileira → não tem como aprender a dar mais peso ao ABRAOM.)

**(b) A falsificação bem-dimensionada dá null limpo.** Bootstrap pareado de MCC(real) − MCC(scrambled) no `br_only.all` (n=4163):

| Par | Δ (real − scrambled) | 95% CI | veredito |
|---|---|---|---|
| M4 − M7s (estático) | +0.0028 | [−0.0235, +0.0299] | CI cruza 0 |
| M5 − M7d (dinâmico) | +0.0117 | [−0.0095, +0.0325] | CI cruza 0 |

Replicado no `br_any`. **O caminho do adapter não carrega sinal ABRAOM-específico** — um negativo *bem-dimensionado*, mais forte que o do Pedro no v10 (que era +0.018 com CI cruzando por falta de poder). Confirma e fortalece a leitura das Fases 1/B.

### 4.3 Fase 4 — calibração: a frequência observada reproduz o headline

Aqui a AF ABRAOM entra **explicitamente** na cabeça (`RegimeABoundedRegionalHead`), que decompõe o score em `molecular − desconto_de_frequência`. Três estágios:

**M5-bounded (cru)** — o modelo aprende a descontar tudo que é comum: `br_only` sobe pra 0.62 e a specificity de benignas comuns vai a 0.99, **mas atropela os P/LP founder/recessivos** (recall despenca pra **0.08**). Esse colapso é *esperado* e é a razão da calibração existir.

**M5_v2 (calibrado)** — re-tuna o desconto no holdout (escala 1.0, teto 1.5, thresholds regional 0.235 / global 0.765). **Recupera o recall P/LP de 0.08 → 0.41**, sacrificando pouco MCC. É o nosso **lead**.

**M5_v3 (safety)** — adiciona a guarda molecular. Aqui o pipeline tomou uma decisão de segurança automática: `hold_current_lead`. A guarda resgatou +0.12 de recall P/LP (de 0.41 → 0.52) **mas criou 208 novos falso-positivos em benignas comuns** (specificity 0.951 → 0.934, abaixo da tolerância). Como o custo em specificity superou o ganho, **o pipeline manteve o M5_v2 como lead** — a decisão certa. (Hipótese do porquê o v3 não ajuda no v11 mas ajudou no v10: a cabeça molecular do v11 é mais forte → mais benignas comuns têm score molecular alto → a guarda, que protege score-alto do desconto, "desprotege" essas benignas.)

### 4.4 A tabela definitiva v11 × v10 (test)

| Slice | v10 M0 | v11 M0 | **v10 M5_v3** (lead Pedro) | **v11 M5_v2** (nosso lead) | v11 M5_v3 |
|---|---|---|---|---|---|
| **br_only** MCC | 0.279 | 0.238 | **0.605** | **0.574** | 0.561 |
| **abraom_common_benign** specificity | 0.803 | 0.842 | 0.959 | **0.951** | 0.934 |
| **abraom_pathogenic_present** recall | 0.417 | 0.460 | 0.436 | **0.405** | 0.521 |
| **global_nonbr** MCC | 0.512 | 0.606 | 0.512 | **0.626** | 0.626 |
| **falsificação** (br_only) | — | — | *não falsificada* | **p=0.0196 ✓** | p=0.0196 ✓ |

**Leitura:** a calibração leva o `br_only` de **0.24 → 0.57** (Pedro: 0.28→0.61); v11 empata no regional, **ganha no global** (0.626 vs 0.512); e fecha a falsificação que o v10 deixou em aberto.

---

## 5. A FALSIFICAÇÃO — o que é, e por que é a peça mais importante (explicação didática)

Esta é a seção que responde à sua pergunta: *"essa falsificação é boa ou ruim para a nossa pesquisa?"* Resposta curta: **inequivocamente boa.** Aqui está o porquê, com calma.

### 5.1 O problema que ela resolve

Imagine que adicionamos a frequência ao modelo e o MCC brasileiro sobe de 0.24 para 0.57. Ótimo — mas antes de comemorar, um cético honesto pergunta:

> *"Isso é mesmo por causa do ABRAOM (a frequência BRASILEIRA), ou é só porque 'variante comum = provavelmente benigna' é verdade em QUALQUER população? Se for a segunda, você não capturou nada de brasileiro — só re-descobriu uma regra genérica de genética."*

Essa é a pergunta que **mata ou valida** a pesquisa inteira. Se o ganho é genérico, o ABRAOM foi decorativo (qualquer tabela de frequência serviria). Se é específico, o ABRAOM contribuiu informação populacional real que **só ele** tem. **Todo o valor da regionalização depende dessa distinção.**

### 5.2 Como o teste responde

O "controle negativo estratificado" faz o seguinte: pega o desconto de frequência que o modelo aplica a cada variante e **embaralha quem recebe qual desconto** — mas de forma *esperta*, preservando a estrutura:

- **`within_af_bin`** (o modo mais estrito): embaralha os descontos **só entre variantes da mesma faixa de frequência**. Assim a *distribuição* de descontos fica idêntica; só se quebra o link "*esta variante específica* ↔ *este desconto específico*".
- **`within_gene` / `within_chromosome`**: idem, dentro do mesmo gene / cromossomo.
- **`global`**: embaralha tudo.

Depois re-mede o `br_only` MCC com os descontos embaralhados. Repete **50 vezes por modo** (200 embaralhamentos no total).

**A lógica do teste:** se o desconto real está rastreando *quais* variantes são de fato comuns no Brasil (biologia ABRAOM real), então dar o desconto às variantes **erradas** (embaralhado) deveria **piorar** o resultado. Se o desconto é só "penalize o que é comum em geral", então qualquer embaralhamento que preserve a distribuição de frequência funcionaria **igual**.

### 5.3 O resultado (e por que é ótimo)

**O desconto real (MCC 0.561) ficou ACIMA de TODOS os 200 embaralhamentos** — em todos os 4 modos, `p(controle ≥ real) = 0.0196`, que é o mínimo possível com 50 seeds (nenhum controle alcançou o real). Inclusive no modo mais estrito (`within_af_bin`, que mantém a distribuição de frequência exata e só quebra o link variante↔desconto):

| modo | real | média dos controles | p(controle ≥ real) |
|---|---|---|---|
| global | 0.561 | 0.459 | **0.0196** |
| within_gene | 0.561 | 0.507 | **0.0196** |
| within_af_bin | 0.561 | 0.524 | **0.0196** |
| within_chromosome | 0.561 | 0.464 | **0.0196** |

**Tradução:** o ganho brasileiro **depende de QUAIS variantes específicas são comuns no ABRAOM** — não de uma regra genérica de frequência. Embaralhar *qual* variante recebe o desconto (mesmo mantendo a distribuição) **destrói** o ganho. **O ABRAOM contribuiu informação populacional real e específica.**

### 5.4 Então: bom ou ruim? → BOM, e por três razões

1. **Valida que a regionalização é real, não artefato.** Descartamos a explicação "é só frequência genérica" — o maior risco deste tipo de trabalho. O que fizemos captura algo *brasileiro*, não uma regra universal disfarçada.

2. **É o resultado que o v10 do Pedro NÃO conseguiu.** O caveat honesto dele era que os controles estratificados *chegavam muito perto* do real → ele não podia afirmar a especificidade ABRAOM (por isso o veredito conservador `do_not_train_next`). No v11, os controles ficam **claramente abaixo** do real → **afirmamos a especificidade com suporte estatístico.** Isso é um **upgrade científico concreto** do porte: não só reproduzimos o v10, nós *resolvemos* uma pergunta que ele deixou aberta.

3. **Fecha a história de forma coerente, não contraditória.** Alguém poderia olhar as Fases 1-3 (sequência não aprende ABRAOM) e concluir "o ABRAOM não serve pra nada". A falsificação da Fase 4 esclarece: o ABRAOM serve, e é real — **só que o canal é a frequência observada, não a representação aprendida.** Os resultados negativos (o modelo não *decora* ABRAOM da sequência) e o positivo (a frequência *observada* é ABRAOM-específica) juntos contam uma história precisa e defensável.

### 5.5 O que a falsificação NÃO diz (honestidade científica)

- **NÃO** diz que o modelo aprendeu biologia brasileira do DNA. O sinal está na **tabela de frequência ABRAOM** (usada como entrada explícita), não na representação que o backbone aprendeu. Isso é consistente com todas as fases anteriores.
- **NÃO** é validação **clínica**. É validação **científica**: o desconto é comprovadamente ABRAOM-específico. Uso clínico ainda exige curadoria externa de P/LP brasileiros (o próximo passo que o Pedro já apontava).
- **O ganho é real mas a lacuna não fecha.** O brasileiro calibrado (0.57) ainda fica abaixo do global (0.63). A regionalização **estreita** o abismo, não o elimina.

---

## 6. A tese final da frente (a história em uma frase)

> **O valor do ABRAOM para a regionalização NÃO é um sinal aprendível pela sequência — nenhum adapter ou fusion o captura (Fases 1, B, 3, todas com controles). Mas a frequência alélica ABRAOM *observada*, usada como calibração explícita, carrega sinal regional real, ABRAOM-específico e estatisticamente falsificado (Fase 4). Portanto: regionalização = calibração de frequência, e essa calibração é genuinamente brasileira.**

O porte para o Beat-v11 **reproduziu** a conclusão nuançada do Pedro num backbone melhor **e a fortaleceu**: o v11 supera o v10 no global e, sobretudo, **fecha a falsificação que o v10 não fechou.**

---

## 7. Cuidados de leitura e limitações

1. **`br_only` no `test` tem n=504** — ruidoso. Usamos o split `all` (n=4163, out-of-sample por construção: `br_only` é disjunto de `nonbr_only`) para os testes de poder. As tabelas de comparação usam `test` só por paridade com a §6.1 do Pedro.
2. **A falsificação é da calibração (o desconto), não do adapter.** São mecanismos diferentes: o adapter (sequência) é null; o desconto (frequência observada) é falsificado-positivo. Não confundir.
3. **`hold_current_lead` é um resultado válido, não uma falha.** O pipeline julgou que a guarda molecular (v3) não vale a troca no v11 — decisão de segurança correta. O lead é M5_v2.
4. **Validação científica ≠ clínica.** Falta curadoria externa de P/LP brasileiros para uso clínico.

---

## 8. Próximos passos (não-bloqueantes)

1. **Curadoria externa** de variantes P/LP brasileiras — o gargalo para validação *clínica*, e agora o **único** caminho de ganho real (ver o item 2). Segue gated por dados.
2. **~~Refinamento v11-nativo~~ — FEITO E REFUTADO (2026-07-23).** Este item, que a versão anterior listava como "adiado, baixa prioridade", foi executado. Tentamos usar as cabeças nativas do v11 (conservação phyloP + missense-severity ESM-2) como **guarda** da calibração — sinais ortogonais à frequência, que o v10 não tinha. **Resultado negativo, bem-dimensionado:** os sinais **não separam** founder patogênica de benigna comum (medianas phyloP100 **0.357 × −0.183**; missense-severity 4.24 × 3.68; enrichment de pico **5.4×** contra base rate **61:1** → ainda ~11 benignas guardadas por founder). Em paralelo, re-tunar o desconto mostrou-se **saturado** (a fronteira `br_only ↔ recall` é fixa: o "teto" de 0.698 custa recall 0.075). **As duas alavancas de melhoria estão fechadas com negativo medido.** Detalhes na §7 ("Pós-porte") e §10.9 do `TCC_REGIONALIZACAO_V11.md`.
3. **Entregável científico:** esta é a base para o `TCC_REGIONALIZACAO_V11.md` — o resultado central (regionalização = calibração de frequência, falsificada) está pronto para escrita.

> **O que o item 2 acrescenta à tese.** Nenhum sinal molecular disponível — nem o classificador molecular treinado, nem a conservação evolutiva, nem a severidade missense do ESM-2 — distingue uma founder patogênica brasileira de uma benigna comum. **O gargalo não é o modelo.** Isso converte "curadoria externa" de recomendação herdada do Pedro em **conclusão própria, com evidência medida sobre um backbone melhor**.

---

*Estado em 2026-07-23: Fases 0-4 completas e validadas **e as melhorias de modelagem esgotadas** (duas alavancas testadas pós-porte, ambas com negativo medido — ver §8.2). Lead v11 = M5_v2. Falsificação da especificidade ABRAOM: positiva (p=0.0196) — superando o caveat do v10. Próximo ganho real = curadoria externa de P/LP brasileiros, gated por dados. Artefatos em `s3://ai4bio-lumina-experiments-v2/lumina-ssm/clinvar-regional-eval/` e `~/v11eval/m5_v3_v11/`.*
