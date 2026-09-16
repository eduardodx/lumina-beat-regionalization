# Runbook — G1 e G2: dados do estudo brasileiro e snapshot de treino da cabeça

Plano: `docs/proposta_mosaic_regionalizacao_desenvolvimento.md` §4.1, §4.2 e §8.
Roda no notebook SageMaker, no `python3` do conda (pandas + pyarrow + pyyaml). **Sem GPU, sem `uv sync`.**
Todos os passos são somente-leitura sobre o release; só escrevem em `--out-dir`.

```bash
export WORK=~/testeArq/lumina-beat-regionalization
cd "$WORK" && git pull
```

## 0. Testes primeiro (teste pulado conta como falha)

```bash
cd "$WORK" && set -euo pipefail && for t in import_mosaic_brazil_studies build_core_locus_head_snapshot build_broad_brazilian_variant_list; do echo "== $t"; REQUIRE_NO_SKIP=1 PYTHONPATH=. python3 tests/test_$t.py; done && echo "TODOS OS TESTES PASSARAM"
```

`set -e` faz o loop parar no primeiro teste que falhar e o shell sair com erro — sem `|| echo`, que mascararia a
falha. Só siga para o passo 1 se aparecer `TODOS OS TESTES PASSARAM`.

Os dois testes de release real procuram `~/mosaic-v1` (ou `MOSAIC_RELEASE`). O de `core_locus` confere os
números do guia para `run_id=0` (196.096 / 2.453 / 1.758, antes das exclusões) com tolerância de 1%: se ele falhar,
**pare** — ou o release mudou, ou a leitura dos folds está errada.

## 1. G1 — importar e validar o membership

```bash
set -o pipefail
PYTHONPATH="$WORK" python3 "$WORK"/scripts/import_mosaic_brazil_studies.py \
    --release-root ~/mosaic-v1 \
    --out-dir ~/artifacts/redesenho/g1_brazil_studies | tee ~/g1.log; echo "exit=$?"
```

Sai `brazil_study_variants.parquet` (membership + `chrom/pos_1based/ref/alt`) e o relatório com as contagens por
estudo, papel, painel e rótulo. Código 2 = alguma promessa do protocolo não se confirmou; nada é publicado.
Conferir no relatório: 8.875 linhas, `br_clinical_evidence` com 3.119 casos (3.116 `case` + 3 `unmatched_case`) e
3.116 controles; `br_population_observed` com 1.889 casos (751 + 1.138) e 751 controles.

## 2. Regra ampla brasileira (insumo do G2 e da §6.4)

Lê o `submission_summary` inteiro: leva alguns minutos.

```bash
set -o pipefail
PYTHONPATH="$WORK" python3 "$WORK"/scripts/build_broad_brazilian_variant_list.py \
    --mosaic-root ~/testeArq/lumina-mosaic \
    --submission-summary ~/clinvar/2026-06/submission_summary_2026-06.txt.gz \
    --pb-examples ~/mosaic-v1/pb_examples.parquet \
    --membership ~/mosaic-v1/studies/brazil/membership.parquet \
    --out-dir ~/artifacts/redesenho/g2_regra_ampla | tee ~/regra_ampla.log; echo "exit=$?"
```

Saem `broad_brazilian_variant_ids.txt`, o manifesto `...txt.manifest.json` (com o sha256 do `pb_examples` do
release usado) e a contagem pré-declarada de **controles do estudo clínico com SCV brasileira**. Se alguma variante
com `br_lab_any` ficar fora da lista, o script **falha com código 2 e não publica** — o G2 recusa a lista de
qualquer modo, porque refaz essa checagem contra o release.

## 3. G2 — snapshot de treino da cabeça

```bash
set -o pipefail
PYTHONPATH="$WORK" python3 "$WORK"/scripts/build_core_locus_head_snapshot.py \
    --release-root ~/mosaic-v1 \
    --brazil-variants ~/artifacts/redesenho/g1_brazil_studies/brazil_study_variants.parquet \
    --broad-br-variant-ids ~/artifacts/redesenho/g2_regra_ampla/broad_brazilian_variant_ids.txt \
    --run-id 0 \
    --out-dir ~/artifacts/redesenho/g2_core_snapshot | tee ~/g2.log; echo "exit=$?"
```

Sai `core_head_snapshot.parquet` (colunas `variant_id`, `role`, rótulo, tier, painel, cluster, fold, coordenadas) e
o relatório com o custo de cada exclusão e o hash lógico que vai no manifesto.

O `--brazil-variants` **não** substitui o membership: as exclusões saem sempre do membership do release, e a saída
do G1 é conferida contra ele (diferença de variantes ou de clusters faz o G2 parar com código 2).

**Política de cluster (`--cluster-exclusion`).** Os *membros* dos estudos saem sempre dos três recortes; a opção
decide o que fazer com os **vizinhos de cluster** deles: `todos` (padrão), `treino` ou `nenhum`. Medido no release
real em 16/09: os golds da validação vivem em 38 clusters e os do teste em 31, e o `br_population_observed` é gold
(~25% do gold do release), então `todos` **zera a validação** — a primeira execução parou com 1 missense e nenhum
splice ou noncoding. O relatório registra, em qualquer política, o custo que `todos` teria
(`custo_potencial_se_todos`), para a escolha ser feita com número e declarada antes de treinar.

Quando o gate reprova, o **relatório sai assim mesmo** (com `status: FALHOU`), porque é nele que estão os números;
o que não é publicado é o snapshot.

O que olhar no relatório:

- `pronto_para_congelar: true` — só sai `true` com a lista da regra ampla **validada**: não vazia, superconjunto do
  `br_lab_any` do release e com manifesto que casa o `pb_examples` **e o sha256 da própria lista**. Ele significa
  "as entradas são válidas e as checagens da política escolhida passaram", **não** que a política de isolamento
  por locus foi aprovada — essa decisão é científica e fica fora do script;
- `identidade`: `hash_composicao` (quem está em cada recorte), `hash_conteudo` (todas as colunas) e
  `arquivo_sha256`. O primeiro não muda se um rótulo mudar — por isso os três;
- `depois_das_exclusoes`: quanto sobrou em cada papel e quantos clusters;
- `exclusoes`: quanto cada uma custou, separado por papel e por classe;
- `por_painel_rotulo` da validação: as duas classes em missense, splice e noncoding — é o que sustenta a seleção.

O chr8 sai por padrão (decisão E ainda pendente). Para medir o custo de mantê-lo, rodar uma segunda vez com
`--no-reserve-chr8` e `--out-dir` diferente — **só para medir**, não para treinar.

## 4. Exposição de locus (substitui a exclusão cega de vizinhos)

```bash
export WORK=~/testeArq/lumina-beat-regionalization && set -o pipefail && PYTHONPATH="$WORK" python3 "$WORK"/scripts/measure_study_locus_exposure.py --snapshot ~/artifacts/redesenho/g2_core_snapshot_nenhum/core_head_snapshot.parquet --brazil-variants ~/artifacts/redesenho/g1_brazil_studies/brazil_study_variants.parquet --out-dir ~/artifacts/redesenho/g2_exposicao 2>&1 | tee ~/exposicao.log; echo "exit=${PIPESTATUS[0]}"
```

Ler `maior`, `menor` e `empate` juntos: empate perfeito dá `fracao_caso_maior = 0`, que é igualdade e não
assimetria. Onde há diferença, olhar `fracao_caso_maior_entre_diferentes` e a magnitude (`mediana_abs`), por
rótulo e por painel. O bloco `por_janela` vale mais que o `por_cluster`: componente conectado encadeia variantes
distantes, e compartilhar cluster não é compartilhar a janela de 4.096 bp que o modelo lê.

Assimetria é risco a declarar, não prova de contaminação; e simetria **não** garante que a exposição seja
inofensiva, porque M0 e MR podem aproveitar os mesmos loci de formas diferentes.

## Regras que não mudam

- O **fold 0 não seleciona nada**: nem extração, nem época, nem Platt, nem limiar. Só avalia depois de congelado.
- Nunca trocar de `run_id` depois de ver resultado de modelo. Se a validação não sustentar a seleção, resolver
  antes de treinar e registrar o motivo.
- Não recompor o pareamento do Mosaic.
- `git add` sempre com caminho explícito; `token/` nunca entra.
