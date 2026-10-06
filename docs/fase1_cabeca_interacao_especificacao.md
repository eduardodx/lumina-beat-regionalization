# Fase 1, cabeça com interação: especificação (rascunho para revisão)

Data: 06/10/2026. É desenvolvimento exploratório, posterior ao teste da Fase 1: os testes já foram vistos.
- **Escopo:** só CPU, sem nova extração. Não autoriza nenhuma corrida de GPU.
- **Estado:** rascunho para revisão antes de implementar. Depois de revisado, fica congelado; o runbook registra o
  hash dele no log antes de rodar.
- **Contexto:** [fase1_br2_resultado.md](fase1_br2_resultado.md), em especial o caso a caso de 06/10.

## Pergunta

A frequência brasileira pode ajudar a reconhecer benignas sem rebaixar automaticamente uma variante cujo sinal
funcional aponta para patogenicidade?

## Por que esta pergunta (leitura posterior do caso a caso)

- **E chama todas.** As 12 P-BR que E+F+BR2 perde em relação a E+F são chamadas por E sozinho. O fpr exigido mediano
  de E é 0,018 nas 9 mantidas e 0,001 nas 3 novas.
- **Já estavam perto do limiar em E+F**, com fpr exigido entre 0,042 e 0,066. A frequência global as rebaixa, e o
  bloco regional completa.
- **A cabeça linear não modula a frequência.** Ela soma a frequência com o mesmo peso para qualquer força do sinal
  funcional.

## Braços

| braço | entradas da cabeça | origem |
|---|---|---|
| E+F | E + F | passo 3 (referência) |
| E+F+BR2 | E + F + BR2 | BR v2 (referência) |
| H(E+F) | E + F + s×F | novo |
| H(E+F+BR2) | E + F + BR2 + s×F + s×BR2 | novo |

- **s:** o escore de E (logit), o mesmo que entra no S+F. Nenhum ajuste novo de E.
  - No treino de cada execução: o escore fora da amostra pela partição interna do passo 3, com purga de 4.096 bp. Já
    está gravado em `diagnosticos/s_fora_da_amostra.parquet`.
  - Na validation e no teste: o escore do braço E do passo 3.
- **s×X:** o produto de s, padronizado no treino, por cada coluna de X, padronizada no treino. Depois, como todas as
  colunas, é padronizado no treino.
  - s×F tem 7 colunas e s×BR2 tem 13.
  - s não entra sozinho, porque já está no espaço de E.
- **BR2:** o bloco do BR v2 sem mudança, inclusive sem a correção de qualidade abaixo. Assim, E+F+BR2 → H(E+F+BR2)
  isola a interação.
- **Mesma família de cabeça:** H(E+F) é o braço sem ABraOM com a mesma família. Ele separa o efeito da informação
  regional do efeito de aumentar a capacidade da cabeça.
- **Mesma receita:** regressão logística L2, grade de C, escolha pela macro AUROC da validation, regra revisada de
  convergência, mesmas linhas, purgas e padronização no treino.

## Hipótese de cada termo

| termo | hipótese |
|---|---|
| s×F | Com sinal funcional forte, a frequência global deveria pesar menos. Exemplo: um alelo recessivo com frequência de portador. |
| s×BR2 | O mesmo para a frequência brasileira |

Risco: a interação também pode poupar benignas com sinal funcional alto e frequência apreciável, o que aumenta os
falso-positivos. Por isso as leituras de benignas abaixo.

## Leituras

Pares:
- **Capacidade, sem ABraOM:** E+F → H(E+F).
- **Principal, a informação regional com a cabeça com interação:** H(E+F) → H(E+F+BR2).
- **A interação, com o BR2 presente:** E+F+BR2 → H(E+F+BR2).
- **O sistema novo contra a referência:** E+F → H(E+F+BR2).
- **Reprodução:** E+F → E+F+BR2.

Mesmo código das leituras do BR v2:
- núcleo: contraste oficial e AUROC por painel;
- proxies: coorte, pareados, controles e interação;
- benefício: AUROC, sensibilidade e falso-positivos entre as benignas presentes no ABraOM;
- P-BR no limiar MCC e na especificidade equivalente, com perdas e ganhos separados;
- as 12 perdas do BR2, uma a uma, com o caso a caso;
- as 13 críticas;
- os coeficientes das interações, só como leitura;
- o avaliador oficial nos braços novos.

## Leitura declarada antes de rodar

É orientação de desenvolvimento, não critério de avanço demonstrado. Cada número vale só para a leitura em que
aparece.

**Apoio, se valerem todos:**
1. **Perdas regionais.** P-BR perdidas por H(E+F) → H(E+F+BR2):
   - no limiar MCC, no máximo 6, metade das 12 de E+F → E+F+BR2;
   - na especificidade equivalente, no máximo 5, metade das 10 de E+F → E+F+BR2 arredondada para baixo.
2. **Proteção das perdas do BR2.** Das 12 P-BR perdidas por E+F → E+F+BR2, H(E+F+BR2) chama pelo menos 6 no seu
   limiar MCC.
3. **Recortes brasileiros.** Os pontos do Δ AUROC de H(E+F) → H(E+F+BR2) não ficam abaixo dos de E+F → E+F+BR2:
   - +0,0015 no proxy clínico;
   - +0,0038 no recorte de benefício, que tem as mesmas variantes do proxy populacional completo.
4. **Benignas presentes no ABraOM.** No recorte de benefício, no limiar MCC, H(E+F+BR2) não tem mais falso-positivos
   que H(E+F).

**Contra, se valer qualquer um:**
- H(E+F) → H(E+F+BR2) perde 10 ou mais P-BR no limiar MCC;
- H(E+F+BR2) chama no máximo 3 das 12 perdas do BR2.

**Fora desses casos:** inconclusivo.

**Em qualquer caso:**
- perdas e ganhos são relatados separados;
- a regra de segurança não muda (margem 0,01, que tolera no máximo uma perda);
- benefício brasileiro exige interação acima dos controles;
- E+F → H(E+F) é lido à parte, como efeito de capacidade, sem entrar no veredito.

## Insumos ausentes e notas

- **Herança por gene (AR/AD) e teto de frequência por gene fora do CSpec.** Não estão no release. As 12 perdas do BR2
  estão em genes de doença recessiva (pela literatura), o que sugere um modulador por gene. Testá-lo exige uma
  fonte nova; nada é baixado agora.
- **Qualidade no BR2.** As faixas de AC e a AF não dependem de PASS. CYP21A2 p.Gln319Ter (15 cópias, VQSR 99,90–100)
  conta como "≥ 10 cópias". A correção fica para uma ablação separada, para não confundir com a interação.

## Consequência

- **Com apoio:** a frequência, global ou brasileira, passa a ser lida condicionada à evidência funcional. Isso entra
  na proposta como princípio da cabeça.
  - Como é desenvolvimento sobre um teste já visto, a confirmação exige dados ainda não vistos (Fase 1b ou outro
    release).
- **Contra:** o sinal funcional não basta para proteger esses alelos. O candidato seguinte é o contexto do gene
  (herança ou teto por gene), que depende de insumo novo e é decisão com o Eduardo.
- **Inconclusivo:** fazer o caso a caso das perdas e ganhos do braço novo antes de qualquer outra mudança.

Nada vai para GPU sem nova decisão.
