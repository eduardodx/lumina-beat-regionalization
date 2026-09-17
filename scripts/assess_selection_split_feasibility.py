#!/usr/bin/env python3
"""Viabilidade de um conjunto de selecao reservado por cluster, recortado do treino -- sem treinar nada.

Roda no notebook (pandas + pyarrow; sem GPU). So le o snapshot do G2; so escreve em --out-dir.
Plano: docs/proposta_mosaic_regionalizacao_desenvolvimento.md, secoes 4.2 e 5.2 (escolha da extracao, gate G5).

POR QUE EXISTE
--------------
A validacao oficial (fold 1, so gold) sobreviveu as exclusoes com 1.575 variantes, mas o suporte onde a selecao
decide e magro: 432 P / 77 B em missense, 93 P / 10 B em splice, 31 P / 19 B em noncoding -- tudo dentro de 38
clusters. Escolher a extracao (G5) com 10 benignas de splice e frageil.

A alternativa e reservar clusters dos folds de TREINO para selecao. Ela so e melhor se tiver suporte suficiente
POR PAINEL E POR CLASSE -- ter mais clusters no total nao garante isso -- e ela nao e de graca: o que for
reservado sai do treino de TODOS os candidatos, entao o custo tem de ser medido junto.

COMO
----
1. Le o snapshot do G2 e mede o teto: quanto ha no treino por painel e classe, e em quantos clusters.
2. Mede a CONCENTRACAO: qual fracao de cada celula (painel, classe) mora nos maiores clusters. Se as benignas de
   splice vivem em 3 clusters, reservar selecao tira essas benignas do treino -- e uma troca, nao um ganho.
3. Recorta clusters de forma gulosa e deterministica ate atingir o alvo por celula, e reporta o que sobra no
   treino. Sem modelo, sem score, sem seed aleatoria.
4. Compara lado a lado com a validacao oficial que ja esta no snapshot.

O QUE NAO PROVA
---------------
- Suporte suficiente nao garante selecao confiavel: a unidade independente continua sendo o cluster, e um recorte
  com muitos exemplos em poucos clusters vale menos do que o n sugere.
- Nao decide a politica: produz os numeros para a decisao ser declarada antes de treinar.
- Nao mede vazamento: os membros dos estudos ja sairam no G2.

USO (notebook)
--------------
    set -o pipefail
    export WORK=~/testeArq/lumina-beat-regionalization
    PYTHONPATH="$WORK" python3 "$WORK"/scripts/assess_selection_split_feasibility.py \
        --snapshot ~/artifacts/redesenho/g2_core_snapshot_nenhum/core_head_snapshot.parquet \
        --min-por-celula 150 \
        --out-dir ~/artifacts/redesenho/g5_selecao | tee ~/selecao.log
    echo "exit=$?"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_core_locus_head_snapshot import (  # noqa: E402
    DISCRIMINATION_PANELS,
    ROLE_TRAIN,
    ROLE_VALIDATION,
)
from scripts.import_mosaic_brazil_studies import counts_by  # noqa: E402

CELL_LABELS = (1, 0)  # P, B


def cell_name(panel: str, label: int) -> str:
    return f"{panel}/{'P' if int(label) == 1 else 'B'}"


def support_by_cell(rows: pd.DataFrame) -> dict[str, dict[str, int]]:
    """Por (painel, classe): quantos exemplos e em quantos clusters distintos."""
    out: dict[str, dict[str, int]] = {}
    for panel in DISCRIMINATION_PANELS:
        for label in CELL_LABELS:
            cell = rows[(rows["primary_panel"].astype(str) == panel)
                        & (rows["binary_label"].astype("Int64") == label)]
            out[cell_name(panel, label)] = {
                "n": int(len(cell)),
                "clusters": int(cell["overlap_cluster_id"].nunique()) if len(cell) else 0,
            }
    return out


def concentration(rows: pd.DataFrame, *, top: int = 5) -> dict[str, Any]:
    """Fracao de cada celula que mora nos maiores clusters: mede se o recorte e uma troca cara."""
    out: dict[str, Any] = {}
    for panel in DISCRIMINATION_PANELS:
        for label in CELL_LABELS:
            cell = rows[(rows["primary_panel"].astype(str) == panel)
                        & (rows["binary_label"].astype("Int64") == label)]
            if cell.empty:
                out[cell_name(panel, label)] = {"n": 0}
                continue
            sizes = cell.groupby("overlap_cluster_id").size().sort_values(ascending=False)
            out[cell_name(panel, label)] = {
                "n": int(len(cell)),
                "clusters": int(len(sizes)),
                "maior_cluster": int(sizes.iloc[0]),
                f"fracao_nos_{top}_maiores": round(float(sizes.head(top).sum() / len(cell)), 4),
                "clusters_para_80pc": int((sizes.cumsum() / len(cell) < 0.8).sum() + 1),
            }
    return out


def cluster_cell_matrix(rows: pd.DataFrame) -> pd.DataFrame:
    """Uma linha por cluster, uma coluna por celula (painel/classe)."""
    frame = rows.copy()
    frame["celula"] = [
        cell_name(str(panel), int(label))
        for panel, label in zip(frame["primary_panel"], frame["binary_label"])
    ]
    cells = [cell_name(p, l) for p in DISCRIMINATION_PANELS for l in CELL_LABELS]
    frame = frame[frame["celula"].isin(cells)]
    if frame.empty:
        return pd.DataFrame(columns=["overlap_cluster_id", *cells]).set_index("overlap_cluster_id")
    matrix = frame.pivot_table(index="overlap_cluster_id", columns="celula", aggfunc="size", fill_value=0)
    for cell in cells:
        if cell not in matrix.columns:
            matrix[cell] = 0
    return matrix[cells]


def greedy_carve(
    matrix: pd.DataFrame, *, target: int, min_clusters: int = 1, cluster_sizes: pd.Series | None = None
) -> tuple[list[str], dict[str, int], dict[str, int]]:
    """Escolhe clusters ate cada celula ter `target` exemplos E `min_clusters` clusters distintos.

    A unidade independente e o cluster: um recorte com 12 mil exemplos em 5 clusters vale menos, para selecao,
    que um com mil exemplos em 200. Por isso o criterio tem as duas partes e o score e CUSTO-CIENTE --
    contribuicao ao deficit dividida pelo tamanho do cluster --, o que prefere muitos clusters pequenos a poucos
    gigantes. Deterministico: empate pelo id do cluster.
    """
    need = {cell: target for cell in matrix.columns}
    have_clusters = {cell: 0 for cell in matrix.columns}
    got = {cell: 0 for cell in matrix.columns}
    chosen: list[str] = []
    remaining = matrix.copy()
    # O custo e o TOTAL de linhas de treino que o cluster leva -- plof, synonymous e other inclusive --, nao so
    # as celulas de discriminacao: um cluster barato nos tres paineis pode arrastar milhares de outros exemplos.
    sizes = (matrix.sum(axis=1) if cluster_sizes is None else cluster_sizes.reindex(matrix.index)).fillna(0)
    sizes = sizes.clip(lower=1)

    def unsatisfied() -> list[str]:
        return [cell for cell in matrix.columns
                if need[cell] > 0 or have_clusters[cell] < min_clusters]

    while unsatisfied() and len(remaining):
        cells = unsatisfied()
        contribution = remaining[cells].clip(upper=1_000_000).gt(0).astype(int).sum(axis=1)
        deficit = remaining[cells].clip(upper=pd.Series({c: max(need[c], 1) for c in cells})[cells],
                                        axis=1).sum(axis=1)
        scores = (deficit + contribution) / sizes.reindex(remaining.index)
        best = scores.max()
        if best <= 0:
            break  # nenhum cluster restante ajuda no que falta
        cluster = sorted(scores[scores == best].index)[0]
        chosen.append(str(cluster))
        for cell in matrix.columns:
            value = int(remaining.loc[cluster, cell])
            got[cell] += value
            need[cell] = max(0, need[cell] - value)
            if value > 0:
                have_clusters[cell] += 1
        remaining = remaining.drop(index=cluster)
    return chosen, got, have_clusters


def build_report(snapshot: pd.DataFrame, *, target: int, top: int, min_clusters: int = 1) -> dict[str, Any]:
    train = snapshot[snapshot["role"] == ROLE_TRAIN]
    validation = snapshot[snapshot["role"] == ROLE_VALIDATION]

    matrix = cluster_cell_matrix(train)
    cluster_sizes = train.groupby("overlap_cluster_id").size()
    chosen, got, cells_clusters = greedy_carve(matrix, target=target, min_clusters=min_clusters,
                                               cluster_sizes=cluster_sizes)
    carved = train[train["overlap_cluster_id"].isin(chosen)]
    rest = train[~train["overlap_cluster_id"].isin(chosen)]

    atingiu = {cell: bool(got[cell] >= target and cells_clusters[cell] >= min_clusters)
               for cell in matrix.columns}
    return {
        "alvo_por_celula": target,
        "min_clusters_por_celula": min_clusters,
        "validacao_oficial_fold1": {
            "n": int(len(validation)),
            "clusters": int(validation["overlap_cluster_id"].nunique()),
            "suporte": support_by_cell(validation),
            "concentracao": concentration(validation, top=top),
            "por_tier": counts_by(validation, ["label_tier"]),
        },
        "como_comparar": (
            "celula a celula: suporte, numero de clusters, concentracao e tier. O total de clusters do conjunto "
            "nao ordena as opcoes -- o fold 1 tem 38 clusters no total mas so 2 com benignas de splice."
        ),
        "teto_no_treino": {
            "n": int(len(train)),
            "clusters": int(train["overlap_cluster_id"].nunique()),
            "suporte": support_by_cell(train),
        },
        "concentracao_no_treino": concentration(train, top=top),
        "recorte_proposto": {
            "clusters": len(chosen),
            "n": int(len(carved)),
            "suporte": support_by_cell(carved),
            "concentracao": concentration(carved, top=top),
            "por_tier": counts_by(carved, ["label_tier"]),
            "atingiu_o_alvo": atingiu,
            "celulas_nao_atingidas": [cell for cell, ok in atingiu.items() if not ok],
        },
        "custo_no_treino": {
            "n_depois": int(len(rest)),
            "clusters_depois": int(rest["overlap_cluster_id"].nunique()),
            "P_depois": int((rest["binary_label"].astype("Int64") == 1).sum()),
            "B_depois": int((rest["binary_label"].astype("Int64") == 0).sum()),
            "suporte_depois": support_by_cell(rest),
            "por_painel_rotulo_reservado": counts_by(carved, ["primary_panel", "binary_label"]),
            "por_tier_depois": counts_by(rest, ["label_tier"]),
        },
        "clusters_reservados": chosen,
        "o_que_nao_prova": [
            "suporte suficiente nao garante selecao confiavel: a unidade independente continua sendo o cluster, "
            "e por isso o criterio exige tambem um minimo de clusters por celula -- que tampouco impede que quase "
            "tudo esteja num cluster so, por isso a concentracao do recorte tambem e publicada",
            "alvos de exemplos e clusters sao parametros exploratorios, nao garantia de selecao confiavel",
            "o recorte sai do treino de TODOS os candidatos: e troca, nao ganho",
            "nao decide a politica; produz numeros para declara-la antes de treinar",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--snapshot", required=True, type=Path, help="core_head_snapshot.parquet do G2")
    parser.add_argument("--min-por-celula", type=int, default=150,
                        help="alvo de exemplos por painel de discriminacao e classe no recorte de selecao")
    parser.add_argument("--min-clusters-por-celula", type=int, default=20,
                        help="minimo de clusters distintos por celula no recorte: a unidade independente e o "
                             "cluster, entao suporte concentrado em poucos clusters nao sustenta selecao")
    parser.add_argument("--top-clusters", type=int, default=5)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args(argv)

    snapshot = pd.read_parquet(args.snapshot.expanduser(),
                               columns=["variant_id", "role", "binary_label", "primary_panel",
                                        "overlap_cluster_id", "label_tier"])
    if not (snapshot["role"] == ROLE_TRAIN).any():
        print("FALHOU: o snapshot nao tem linhas de treino.")
        return 2

    report = build_report(snapshot, target=args.min_por_celula, top=args.top_clusters,
                          min_clusters=args.min_clusters_por_celula)
    out_dir = args.out_dir.expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "viabilidade_conjunto_de_selecao.json"
    report["entradas"] = {"snapshot": str(args.snapshot)}
    report["saidas"] = {"relatorio": str(report_path)}
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

    resumo = {key: report[key] for key in
              ("alvo_por_celula", "min_clusters_por_celula", "validacao_oficial_fold1", "recorte_proposto",
               "custo_no_treino")}
    resumo["concentracao_no_treino"] = report["concentracao_no_treino"]
    resumo["saidas"] = report["saidas"]
    print(json.dumps(resumo, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
