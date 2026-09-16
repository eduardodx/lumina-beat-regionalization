#!/usr/bin/env python3
"""Exposicao de locus: quanto do treino divide `overlap_cluster_id` com cada membro dos estudos brasileiros.

Roda no notebook (pandas + pyarrow; sem GPU). So le; so escreve em --out-dir.
Plano: docs/proposta_mosaic_regionalizacao_desenvolvimento.md, secoes 4.2, 6.4 e 7.

POR QUE EXISTE
--------------
Medido em 16/09 no release real: excluir do treino todas as variantes que dividem cluster com algum membro dos
estudos custa 153.663 exemplos (78%) e 28.656 dos 33.897 patogenicos (95%) -- os clusters dos membros cobrem os
genes clinicamente sequenciados, que e onde vivem os patogenicos do ClinVar. A exclusao cega sai mais cara do que
o risco que ela evita, e ainda deixa validacao e teste cheios dos mesmos loci.

A troca declarada e: excluir os MEMBROS (obrigatorio pelo protocolo) e MEDIR a exposicao de locus em vez de
tentar zera-la. O risco real nao e a exposicao em si -- ela e identica para o sistema base e o regionalizado,
que compartilham o snapshot -- e sim a exposicao ser DIFERENTE entre casos e controles, porque o pareamento do
Mosaic e por rotulo, painel e bin de AF, nunca por gene. Uma assimetria ai pode atravessar a interacao.

COMO
----
1. Le o snapshot do G2 e conta, por `overlap_cluster_id`, quantas variantes de TREINO existem (total, P e B).
2. Junta essa contagem a cada membro dos dois estudos (saida do G1).
3. Compara casos pareados com os seus controles, par a par, no estudo clinico: diferenca mediana, fracao de pares
   em que o caso tem mais exposicao, e fracao de membros com exposicao zero.

O QUE NAO PROVA
---------------
- Exposicao potencial nao e memorizacao: dividir cluster com exemplos de treino nao demonstra que o modelo usou
  aquele locus. E um covariavel declarado, nao um veredito.
- Simetria entre casos e controles nao garante ausencia de efeito; assimetria nao prova contaminacao. O numero
  entra no relatorio e, se for grande, vira analise de sensibilidade pre-declarada.
- O cluster do Mosaic e componente conectado numa janela de ate 32 kb: dois membros do mesmo gene costumam cair
  no mesmo cluster, entao a unidade e grosseira de proposito.

USO (notebook)
--------------
    set -o pipefail
    export WORK=~/testeArq/lumina-beat-regionalization
    PYTHONPATH="$WORK" python3 "$WORK"/scripts/measure_study_locus_exposure.py \
        --snapshot ~/artifacts/redesenho/g2_core_snapshot_nenhum/core_head_snapshot.parquet \
        --brazil-variants ~/artifacts/redesenho/g1_brazil_studies/brazil_study_variants.parquet \
        --out-dir ~/artifacts/redesenho/g2_exposicao | tee ~/exposicao.log
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

from scripts.import_mosaic_brazil_studies import (  # noqa: E402
    ROLE_CASE,
    ROLE_CONTROL,
    STUDIES,
    STUDY_CLINICAL,
)

ROLE_TRAIN = "train"


def train_exposure_by_cluster(snapshot: pd.DataFrame) -> pd.DataFrame:
    """Por cluster: quantas variantes de treino existem, e quantas sao P e B."""
    train = snapshot[snapshot["role"] == ROLE_TRAIN]
    labels = train["binary_label"].astype("Int64")
    grouped = pd.DataFrame({
        "overlap_cluster_id": train["overlap_cluster_id"],
        "n_treino": 1,
        "n_treino_P": (labels == 1).astype(int),
        "n_treino_B": (labels == 0).astype(int),
    }).groupby("overlap_cluster_id", as_index=False).sum()
    return grouped


def join_exposure(members: pd.DataFrame, exposure: pd.DataFrame) -> pd.DataFrame:
    joined = members.merge(exposure, on="overlap_cluster_id", how="left")
    for column in ("n_treino", "n_treino_P", "n_treino_B"):
        joined[column] = joined[column].fillna(0).astype(int)
    return joined


def describe(series: pd.Series) -> dict[str, Any]:
    if series.empty:
        return {"n": 0}
    return {
        "n": int(len(series)),
        "zero": int((series == 0).sum()),
        "fracao_zero": round(float((series == 0).mean()), 4),
        "mediana": float(series.median()),
        "q1": float(series.quantile(0.25)),
        "q3": float(series.quantile(0.75)),
        "media": round(float(series.mean()), 2),
        "maximo": int(series.max()),
    }


def paired_comparison(members: pd.DataFrame, study_id: str) -> dict[str, Any]:
    """Caso x seu controle, par a par: e a assimetria que pode atravessar a interacao."""
    rows = members[members["study_id"] == study_id]
    exposure_of = dict(zip(rows["variant_id"], rows["n_treino"]))
    cases = rows[rows["member_role"] == ROLE_CASE]
    pairs = [
        (int(exposure_of[vid]), int(exposure_of[str(mid)]))
        for vid, mid in zip(cases["variant_id"], cases["matched_variant_id"])
        if not pd.isna(mid) and str(mid) in exposure_of
    ]
    if not pairs:
        return {"pares": 0}
    diffs = pd.Series([case - control for case, control in pairs])
    return {
        "pares": len(pairs),
        "caso": describe(pd.Series([c for c, _ in pairs])),
        "controle": describe(pd.Series([c for _, c in pairs])),
        "diferenca_caso_menos_controle": {
            "mediana": float(diffs.median()),
            "media": round(float(diffs.mean()), 2),
            "fracao_caso_maior": round(float((diffs > 0).mean()), 4),
            "fracao_empate": round(float((diffs == 0).mean()), 4),
            "fracao_ambos_zero": round(
                float(sum(1 for c, k in pairs if c == 0 and k == 0) / len(pairs)), 4
            ),
        },
        "leitura": "fracao_caso_maior perto de 0,5 indica exposicao simetrica entre casos e controles; longe "
                   "disso e sinal de risco a declarar, nao prova de contaminacao",
    }


def build_report(members: pd.DataFrame, snapshot: pd.DataFrame, paths: dict[str, str]) -> dict[str, Any]:
    train = snapshot[snapshot["role"] == ROLE_TRAIN]
    report: dict[str, Any] = {
        "entradas": paths,
        "treino": {
            "n": int(len(train)),
            "clusters": int(train["overlap_cluster_id"].nunique()),
        },
        "por_estudo_e_papel": {},
        "pareado": {},
        "o_que_nao_prova": [
            "exposicao potencial nao e memorizacao",
            "a exposicao e a mesma para base e regionalizado: so a assimetria caso x controle pode atravessar "
            "a interacao",
        ],
    }
    for study in STUDIES:
        rows = members[members["study_id"] == study]
        if rows.empty:
            continue
        report["por_estudo_e_papel"][study] = {
            role: describe(rows.loc[rows["member_role"] == role, "n_treino"])
            for role in sorted(set(rows["member_role"]))
        }
    for study in STUDIES:
        if (members["study_id"] == study).any():
            report["pareado"][study] = paired_comparison(members, study)
    report["destaque"] = report["pareado"].get(STUDY_CLINICAL, {})
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--snapshot", required=True, type=Path, help="core_head_snapshot.parquet do G2")
    parser.add_argument("--brazil-variants", required=True, type=Path, help="saida do G1")
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args(argv)

    snapshot = pd.read_parquet(args.snapshot.expanduser(),
                               columns=["variant_id", "role", "binary_label", "overlap_cluster_id"])
    members = pd.read_parquet(args.brazil_variants.expanduser(),
                              columns=["variant_id", "study_id", "member_role", "matched_variant_id",
                                       "binary_label", "primary_panel", "overlap_cluster_id"])
    if not (snapshot["role"] == ROLE_TRAIN).any():
        print("FALHOU: o snapshot nao tem linhas de treino.")
        return 2

    joined = join_exposure(members, train_exposure_by_cluster(snapshot))
    leaked = set(joined["variant_id"]) & set(snapshot["variant_id"])
    if leaked:
        print(f"FALHOU: {len(leaked)} membros dos estudos estao dentro do snapshot; corrija o G2 antes.")
        return 2

    out_dir = args.out_dir.expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    table_path = out_dir / "exposicao_por_membro.parquet"
    report_path = out_dir / "exposicao_de_locus.json"
    joined.to_parquet(table_path, index=False)

    report = build_report(joined, snapshot, {
        "snapshot": str(args.snapshot), "brazil_variants": str(args.brazil_variants),
    })
    report["saidas"] = {"tabela": str(table_path), "relatorio": str(report_path)}
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "treino": report["treino"],
        "clinico_pareado": report["pareado"].get(STUDY_CLINICAL),
        "saidas": report["saidas"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
