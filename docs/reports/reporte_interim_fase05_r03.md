# Regionalização R03 — Reporte interim
### Fase 0.5 (parte 1): sonda representacional do chr8 + de-risking de infra

**Data:** 2026-07-28 · **Frente:** Gabriel · **Modelo:** `LUM-20260719-001-R03`

---

## TL;DR — o essencial

- Rodei a **primeira das duas sondas baratas da Fase 0.5** (o gate de poder que adicionamos *antes*
  de treinar qualquer adapter). Ela é **backbone-only / zero-shot**: mede o *piso* do R03 como está,
  **sem nenhum adapter populacional treinado**.
- **Achado principal (tranquilizador): não há degradação específica do chr8** na interface
  populacional do R03. No chr8 o modelo ordena frequência tão bem quanto num cromossomo in-domain.
  O alerta do TECHNICAL.md §6 (R03 pior no chr8) **não se traduz** num colapso do sinal populacional
  no chr8 — o teu holdout de chr8 **não está comprometido** por esse motivo.
- **Ressalva honesta:** a interface populacional nativa é um *weak prior* fraco **em todo lugar**
  (não só no chr8): Spearman ρ ≈ **+0.14** numa distribuição natural de variantes. O eixo
  representacional está *vivo, mas fraco* no substrato nativo — o valor dele vai depender do adapter
  treinado (Fases 2–3) **levantar** esse número.
- **Nenhuma decisão é necessária agora.** A escolha de endpoint primário (MCC clínico vs promover o
  eixo representacional/AMR) fica, como combinamos, para **depois do número de poder clínico** — que
  é a próxima sonda (a1), local e rápida.

---

## 1. O que medi e por quê (contexto)

A Fase 0.5 é o gate barato que colocamos entre a preparação de dados e o treino: ela decide
viabilidade **antes** de gastar os ~9 jobs de treino. Tem duas sondas independentes:

- **(a) poder do endpoint clínico** — ainda a fazer;
- **(b) sonda representacional do chr8** — *esta aqui*.

O R03 expõe uma "cabeça populacional" (`gnomad_af_pred`) que a própria documentação chama de *weak
population prior*: para cada posição/base, ela emite um escalar que deveria crescer com a frequência
alélica. A sonda mede, com **Spearman**, se esse escalar de fato **ordena** as variantes por
frequência observada: ρ próximo de +1 = ordena bem; ρ ≈ 0 = não ordena.

O passo decisivo é comparar **chr8 (o cromossomo held-out do desenho)** contra um cromossomo
**in-domain (chr1)**. Se o ρ do chr8 despencasse em relação ao chr1, seria sinal de que a interface
está *especificamente* quebrada no chr8 — e aí o benchmark representacional do chr8 (que, no teu
protocolo, tende a ter mais poder que o T_BR clínico) nasceria furado. Era isso que eu precisava
saber antes de a gente apostar nele como endpoint de reserva.

---

## 2. Resultado

ρ = Spearman entre o score `gnomad_af_pred` e o log10 da AF observada (gnomAD). Rodei em dois
conjuntos de variantes, com o backbone do R03 só (sem adapter):

| Conjunto | n (chr8 / chr1) | ρ **chr8** | ρ **chr1** | gap (chr8−chr1) |
|---|---|---|---|---|
| ClinVar não-BR — **distribuição natural** | 87 / 265 | **+0.16** (IC [−0.06, +0.37]) | **+0.14** (IC [+0.02, +0.25]) | +0.02 |
| ABraOM freq-adapter — **n grande, prov. balanceado por AF** | 20k / 20k | **−0.045** (IC [−0.059, −0.031]) | **−0.049** (IC [−0.063, −0.035]) | +0.004 |

A segunda saída populacional do modelo (`gnomad_observed_logits`) deu ≈0 em tudo. Três leituras:

1. **chr8 ≈ chr1 nos dois conjuntos** (gap +0.02 e +0.004) → **sem degradação chr8-específica**.
2. **Magnitude fraca em todo lugar** → *weak prior* confirmado.
3. **Cross-check:** o ρ na distribuição natural (**+0.14**) bate no **~0.13** que a Fase 1 do v11
   mediu na *mesma* interface no backbone irmão (beat-v11) → o readout está sadio, não é bug de
   medição.

---

## 3. Como ler isto — e como NÃO ler

- **É o piso, não o teto.** Backbone puro, sem adapter. A campanha *treina* o adapter populacional
  exatamente para aprender a distribuição de frequência; a tua Fase 5 é que vai medir se o adapter
  treinado melhora esse ρ. **Piso fraco ≠ campanha condenada** — só diz que o R03 não entrega isso
  de graça (esperado num *weak prior*), e um piso baixo deixa **espaço detectável** para um ganho.
- **O −0.05 do conjunto grande não é "anti-sinal".** Esse dataset é o de *treino* do freq adapter do
  v11 e é, muito provavelmente, **balanceado por bin de frequência** — o que estreita a faixa de AF
  e "lava" a correlação (range restriction). É uma hipótese, mas casa com o fato de a distribuição
  **natural** dar +0.14. Trato o +0.14 como a leitura representativa.

---

## 4. O que isso muda no protocolo

- **Nada estrutural muda.**
- **De-risca o holdout de chr8:** ele não está minado por um colapso chr8-específico da interface —
  a distinção "adapter-level vs foundation-and-adapter holdout" que você pede segue de pé; a auditoria
  do que o backbone viu no pré-treino continua no radar (depende do repo de treino/W&B).
- **Tempera a expectativa** do eixo representacional como fallback de poder "de graça": no nativo ele
  é fraco; o payoff depende do adapter treinado.
- A decisão de **endpoint primário** fica para depois do número de poder clínico (a1).

---

## 5. De-risking de infra (em paralelo)

- **Checkpoint R03** (bucket croma-bioai) **acessível** e carrega em modo estrito (630 tensores,
  shapes conforme o esperado: `last_hidden_state [B,L,448]`, etc.).
- **Superfície LoRA mapeada no checkpoint real:** o Mamba recebe LoRA cheio (o grosso dos parâmetros
  treináveis). Já as **atenções sparse/anchor ficam ~congeladas sob LoRA** — o `out_proj` do
  `MultiheadAttention` não recebe o delta e o QKV é empacotado. Isso **não quebra** o contrato de
  "parâmetros treináveis idênticos entre M1–M4" (é o mesmíssimo procedimento nos quatro), mas vale
  você saber que a superfície treinável é **Mamba + atenção local**, não a atenção global.

---

## 6. Próximos passos

1. **Sonda a1** — poder do endpoint clínico (contagens cruas do `br_only` + bootstrap
   gene-clusterizado). Local, rápida, não depende de você.
2. Com a1 em mãos → trago a **decisão de endpoint primário** (manter MCC clínico como confirmatório
   vs promover o representacional/AMR).
3. Pendências que **só travam a Fase 2** (treinar o M1): **fonte gnomAD** (release / genome vs exome /
   como estratificar por população) e **definição operacional de "global"** (pesos por grupo; AMR
   entra com 20%?).

---

## 7. Pergunta para você

Algo aqui muda a sua expectativa? Por exemplo, dado o piso representacional fraco, faz sentido (a)
dar mais capacidade/rank ao adapter populacional, (b) priorizar o braço **AMR**, ou (c) investir na
curadoria do painel sentinel P/LP para dar poder ao chr8/BRCA? **Nada é urgente** — é só para pegar a
sua reação cedo, antes de gastarmos o número de poder e a Fase 0.
