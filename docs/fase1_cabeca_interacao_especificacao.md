# Fase 1, cabeça com interação: especificação, escrita antes de rodar

Data: 06/10/2026. É desenvolvimento exploratório, posterior ao teste da Fase 1: os testes já foram vistos.
- **Escopo:** só CPU, sem nova extração. Não autoriza nenhuma corrida de GPU.
- **Estado:** o rascunho de `ca7ceeb` foi revisado em 06/10, e os três ajustes da revisão estão aplicados. Fica
  congelado para implementar; o runbook registra o hash dele no log antes de rodar.
- **Contexto:** [fase1_br2_resultado.md](fase1_br2_resultado.md), em especial o caso a caso de 06/10.

## Pergunta

A frequência brasileira pode ajudar a reconhecer benignas sem rebaixar automaticamente uma variante cujo sinal
funcional aponta para patogenicidade?

Aqui, "sinal funcional" é o escore aprendido de E, o classificador só com o embedding. É uma previsão, não evidência
funcional experimental. E negativo quer dizer só que esse classificador, nesse limiar, não chamou a variante; o
embedding pode ter informação que outra cabeça aproveite.

## Por que esta pergunta (leitura posterior do caso a caso)

- As 12 P-BR que E+F+BR2 perde em relação a E+F são reconhecidas pelo classificador só com o embedding. Elas deixam de
  ser reconhecidas em configurações que incluem frequência.
- E+F e E+F+BR2 foram reajustados separadamente e têm limiares próprios. Isso motiva o teste, mas não prova que a
  frequência causou as perdas.
- O teste: ver se uma combinação mais flexível preserva essas chamadas.

## Braços

| braço | entradas da cabeça | origem |
|---|---|---|
| E+F | E + F | passo 3 (referência) |
| E+F+BR2 | E + F + BR2 | BR v2 (referência) |
| H(E+F) | E + F + S + S×F | novo |
| H(E+F+BR2) | E + F + BR2 + S + S×F + S×BR2 | novo |

- **S:** o escore de E (logit), o mesmo que entra no S+F. Nenhum ajuste novo de E.
  - No treino de cada execução: o escore fora da amostra pela partição interna do passo 3, com purga de 4.096 bp. Já
    está gravado em `diagnosticos/s_fora_da_amostra.parquet`.
  - Na validation e no teste: o escore do braço E do passo 3.
- **Por que S entra sozinho:** no treino, os escores fora da amostra vêm de três classificadores internos diferentes,
  então não estão no espaço de E.
- **S×X:** o produto de S, padronizado no treino da execução, por cada coluna de X, padronizada no treino. Depois, como
  todas as colunas, a matriz inteira é padronizada no treino. S×F tem 7 colunas e S×BR2 tem 13.
- **Conferência do S reaproveitado, que interrompe a corrida se falhar:**
  - **IDs e execução:** em cada execução, o S de treino tem os IDs das linhas de treino, na mesma ordem, e o fold
    interno de cada linha é o fold dela. O S de validation e de teste tem os IDs desses papéis.
  - **Valores:** média, desvio e n de treino, validation e teste batem com o `s_resumo` do S+F no `selecao.json` do
    passo 3.
  - **Purgas:** a partição interna refeita tem, em cada fold, os mesmos tamanhos de treino e de teste e as mesmas
    purgas do `s_interno` gravado.
  - **C:** os ajustes internos usaram o C do braço E da execução.
- **BR2:** o bloco do BR v2 sem mudança, inclusive sem a correção de qualidade abaixo. Assim, E+F+BR2 → H(E+F+BR2)
  isola a cabeça.
- **Mesma família de cabeça:** H(E+F) é o braço sem ABraOM com a mesma família. Ele separa o efeito da informação
  regional do efeito de aumentar a capacidade da cabeça.
- **Mesma receita:** regressão logística L2, grade de C, escolha pela macro AUROC da validation, regra revisada de
  convergência, mesmas linhas, purgas e padronização no treino.

## Hipótese de cada termo

| termo | hipótese |
|---|---|
| S | Permite a hierarquia dos termos de interação; o efeito principal já está em E |
| S×F | Com escore de E alto, a frequência global deveria pesar menos |
| S×BR2 | O mesmo para a frequência brasileira |

- **Motivação de S×F e S×BR2:** alelos de doença recessiva com frequência de portador. É uma hipótese tirada do caso a
  caso, não uma explicação: a herança precisa ser associada à doença e à variante, não ao gene inteiro.
- **Risco:** a interação também pode poupar benignas com escore de E alto e frequência apreciável, o que aumenta os
  falso-positivos. Por isso as leituras de benignas abaixo.

## Leituras

Pares:
- **Capacidade, sem ABraOM:** E+F → H(E+F). É relatado com perdas e ganhos, porque mostra o que a mudança de cabeça
  sozinha já causa.
- **A informação regional com a cabeça com interação:** H(E+F) → H(E+F+BR2).
- **Segurança do sistema novo contra a referência original:** E+F → H(E+F+BR2).
- **A cabeça com o BR2 presente:** E+F+BR2 → H(E+F+BR2).
- **Reprodução:** E+F → E+F+BR2.

Mesmo código das leituras do BR v2:
- núcleo: contraste oficial e AUROC por painel;
- proxies: coorte, pareados, controles e interação;
- benefício: AUROC, sensibilidade e falso-positivos entre as benignas presentes no ABraOM;
- P-BR no limiar MCC e na especificidade equivalente, com perdas e ganhos separados, sem compensar perdas com ganhos;
- as 12 perdas do BR2, uma a uma;
- as 13 críticas;
- os coeficientes de S e das interações, só como leitura;
- o avaliador oficial nos braços novos, conferindo os limiares.

## Leitura declarada antes de rodar

É orientação de desenvolvimento, não critério de avanço demonstrado. Cada número vale só para a leitura em que
aparece. As perdas são contadas sozinhas, sem descontar ganhos. O script calcula o veredito por estas regras.

**Apoio, se valerem todos:**
1. **Perdas, nos dois contrastes:** H(E+F) → H(E+F+BR2) e E+F → H(E+F+BR2), cada um:
   - no limiar MCC, no máximo 6 P-BR perdidas, metade das 12 de E+F → E+F+BR2;
   - na especificidade equivalente, no máximo 5, metade das 10 de E+F → E+F+BR2 arredondada para baixo.
2. **Proteção das perdas do BR2:** das 12 P-BR perdidas por E+F → E+F+BR2, H(E+F+BR2) chama pelo menos 6 no seu
   limiar MCC.
3. **Recortes brasileiros:** os pontos do Δ AUROC de H(E+F) → H(E+F+BR2) não ficam abaixo dos de E+F → E+F+BR2,
   reproduzidos na mesma corrida:
   - no proxy clínico, coorte completo (+0,0015 no BR v2);
   - no recorte de benefício (+0,0038 no BR v2), que tem as mesmas variantes do proxy populacional completo.
4. **Benignas presentes no ABraOM:** no recorte de benefício, no limiar MCC, H(E+F+BR2) não tem mais
   falso-positivos que H(E+F).

**Contra, se valer qualquer um:**
- um dos dois contrastes do item 1 perde 10 ou mais P-BR no limiar MCC;
- H(E+F+BR2) chama no máximo 3 das 12 perdas do BR2.

**Fora desses casos:** inconclusivo.

**Em qualquer caso:**
- a regra de segurança não muda (margem 0,01, que tolera no máximo uma perda) e é relatada nos dois contrastes;
- benefício brasileiro exige interação acima dos controles;
- se a conferência do S falhar, nada é treinado.

## Insumos ausentes e notas

- **Herança por doença e por variante.** Não está no release. As 12 perdas do BR2 estão em genes com doença recessiva
  descrita, mas isso não basta: há manifestações em heterozigotos em alguns desses genes (por exemplo WNT10A). Um
  modulador por gene ou por variante exige uma fonte nova; nada é baixado agora.
- **Qualidade no BR2.** As faixas de AC e a AF não dependem de PASS: CYP21A2 p.Gln319Ter (15 cópias, VQSR
  99,90–100) conta como "≥ 10 cópias". É uma limitação da receita declarada. A correção fica para uma ablação
  separada, para não se confundir com a cabeça.

## Consequência, limitada ao experimento

- **Com apoio:** evidência de desenvolvimento para esta forma de combinar a frequência com o escore de E. Não é
  regionalização do embedding demonstrada. Como o teste já foi visto, a confirmação exige dados ainda não vistos
  (Fase 1b ou outro release).
- **Contra:** esta interação, nesta receita, não preservou as chamadas. Isso não prova que o embedding não tenha sinal
  suficiente. O candidato seguinte é o contexto da doença e da variante, que depende de insumo novo e é decisão com o
  Eduardo.
- **Inconclusivo:** fazer o caso a caso das perdas e ganhos dos braços novos antes de qualquer outra mudança.

Nada vai para GPU sem nova decisão.
