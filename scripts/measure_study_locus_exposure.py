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
tentar zera-la. Isto e um diagnostico de campanha de desenvolvimento, nao uma demonstracao de que a exposicao e
inofensiva.

COMO
----
1. Le o snapshot do G2 e mede a exposicao de cada membro por duas unidades:
   - `n_treino`: variantes de treino no mesmo `overlap_cluster_id` (unidade do Mosaic, grossa: componente
     conectado em ate 32 kb, que encadeia variantes distantes);
   - `n_janela`: variantes de treino a ate `--radius-bp` do membro, isto e, as que de fato cairiam dentro da
     janela de sequencia que o modelo le. Compartilhar cluster NAO e compartilhar janela.
2. Junta as duas a cada membro dos dois estudos (saida do G1).
3. Compara casos pareados com os seus controles, par a par, reportando caso maior, menor e empate -- e a fracao
   de "caso maior" ENTRE OS PARES DIFERENTES. Estratifica por rotulo e por painel.

O QUE NAO PROVA
---------------
- Exposicao potencial nao e memorizacao: dividir cluster ou janela com exemplos de treino nao demonstra que o
  modelo usou aquele locus. E um covariavel declarado, nao um veredito.
- **Snapshot compartilhado nao faz o risco desaparecer no contraste M0 x MR.** Os dois sistemas veem os mesmos
  loci, mas com representacoes diferentes, e podem aproveita-los de forma diferente: mesma exposicao nao implica
  mesmo efeito. A assimetria entre casos e controles medida aqui e UM mecanismo, nao o unico.
- Simetria nao garante ausencia de efeito, e assimetria nao prova contaminacao. Empate perfeito em todos os pares
  da `fracao_caso_maior = 0` com `fracao_empate = 1`: por isso os tres numeros sao publicados juntos, e a leitura
  correta e pela fracao entre os pares diferentes mais a magnitude das diferencas.
- Sobreposicao de locus entre treino e estudo nao e inevitavel: e uma escolha de custo, medida no G2.

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


def window_exposure(members: pd.DataFrame, snapshot: pd.DataFrame, *, radius_bp: int) -> pd.DataFrame:
    """Variantes de treino a ate `radius_bp` de cada membro: o que de fato cabe na janela que o modelo le.

    O `overlap_cluster_id` e componente conectado em ate 32 kb e encadeia variantes distantes; duas variantes do
    mesmo cluster podem nunca aparecer na mesma janela de 4.096 bp. As duas medidas sao reportadas lado a lado.
    """
    import numpy as np

    train = snapshot[snapshot["role"] == ROLE_TRAIN]
    positions: dict[str, Any] = {}
    positions_p: dict[str, Any] = {}
    for chrom, group in train.groupby("chrom"):
        pos = np.sort(group["pos_1based"].to_numpy(dtype="int64"))
        positions[str(chrom)] = pos
        p_rows = group[group["binary_label"].astype("Int64") == 1]
        positions_p[str(chrom)] = np.sort(p_rows["pos_1based"].to_numpy(dtype="int64"))

    def count(pos_by_chrom, chrom, pos) -> int:
        arr = pos_by_chrom.get(str(chrom))
        if arr is None or not len(arr):
            return 0
        lo = int(np.searchsorted(arr, pos - radius_bp, side="left"))
        hi = int(np.searchsorted(arr, pos + radius_bp, side="right"))
        return hi - lo

    out = members.copy()
    out["n_janela"] = [count(positions, c, p) for c, p in zip(members["chrom"], members["pos_1based"])]
    out["n_janela_P"] = [count(positions_p, c, p) for c, p in zip(members["chrom"], members["pos_1based"])]
    return out


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


def pair_stats(pairs: list[tuple[int, int]]) -> dict[str, Any]:
    """Maior, menor e empate sao publicados juntos: so `fracao_caso_maior` confunde empate com assimetria."""
    if not pairs:
        return {"pares": 0}
    diffs = pd.Series([case - control for case, control in pairs])
    different = diffs[diffs != 0]
    return {
        "pares": len(pairs),
        "caso": describe(pd.Series([c for c, _ in pairs])),
        "controle": describe(pd.Series([c for _, c in pairs])),
        "diferenca_caso_menos_controle": {
            "mediana": float(diffs.median()),
            "media": round(float(diffs.mean()), 2),
            "mediana_abs": float(diffs.abs().median()),
            "fracao_caso_maior": round(float((diffs > 0).mean()), 4),
            "fracao_caso_menor": round(float((diffs < 0).mean()), 4),
            "fracao_empate": round(float((diffs == 0).mean()), 4),
            "pares_diferentes": int(len(different)),
            "fracao_caso_maior_entre_diferentes": (
                round(float((different > 0).mean()), 4) if len(different) else None
            ),
            "fracao_ambos_zero": round(
                float(sum(1 for c, k in pairs if c == 0 and k == 0) / len(pairs)), 4
            ),
        },
    }


def build_pairs(rows: pd.DataFrame, column: str) -> list[tuple[int, int, Any, Any]]:
    """(exposicao do caso, exposicao do controle, rotulo, painel) por par. Rotulo e painel sao iguais no par."""
    exposure_of = dict(zip(rows["variant_id"], rows[column]))
    cases = rows[rows["member_role"] == ROLE_CASE]
    out: list[tuple[int, int, Any, Any]] = []
    for vid, mid, label, panel in zip(cases["variant_id"], cases["matched_variant_id"],
                                      cases["binary_label"], cases["primary_panel"]):
        if pd.isna(mid) or str(mid) not in exposure_of:
            continue
        out.append((int(exposure_of[vid]), int(exposure_of[str(mid)]), label, panel))
    return out


def paired_comparison(members: pd.DataFrame, study_id: str, *, column: str = "n_treino") -> dict[str, Any]:
    """Caso x seu controle, par a par, no total e por rotulo e painel: um resumo global esconde diferencas."""
    rows = members[members["study_id"] == study_id]
    pairs = build_pairs(rows, column)
    if not pairs:
        return {"pares": 0}
    report = pair_stats([(case, control) for case, control, _, _ in pairs])
    report["por_rotulo"] = {
        str(label): pair_stats([(c, k) for c, k, lab, _ in pairs if lab == label])
        for label in sorted({lab for _, _, lab, _ in pairs})
    }
    report["por_painel"] = {
        str(panel): pair_stats([(c, k) for c, k, _, pan in pairs if pan == panel])
        for panel in sorted({str(pan) for _, _, _, pan in pairs})
    }
    report["leitura"] = (
        "publicar maior/menor/empate juntos: empate perfeito da fracao_caso_maior = 0 com fracao_empate = 1, "
        "que e igualdade e nao assimetria. Onde ha diferenca, olhar fracao_caso_maior_entre_diferentes e a "
        "magnitude (mediana_abs). Assimetria e risco a declarar, nao prova de contaminacao -- e simetria nao "
        "garante que a exposicao seja inofensiva, porque M0 e MR podem aproveitar os mesmos loci de formas "
        "diferentes."
    )
    return report


def build_report(
    members: pd.DataFrame, snapshot: pd.DataFrame, paths: dict[str, str], *, radius_bp: int
) -> dict[str, Any]:
    train = snapshot[snapshot["role"] == ROLE_TRAIN]
    report: dict[str, Any] = {
        "entradas": paths | {"radius_bp": radius_bp},
        "treino": {
            "n": int(len(train)),
            "clusters": int(train["overlap_cluster_id"].nunique()),
        },
        "unidades": {
            "n_treino": "variantes de treino no mesmo overlap_cluster_id (componente conectado em ate 32 kb)",
            "n_janela": f"variantes de treino a ate {radius_bp} bp -- o que cabe na janela lida pelo modelo",
        },
        "por_estudo_e_papel": {},
        "pareado_por_cluster": {},
        "pareado_por_janela": {},
        "o_que_nao_prova": [
            "exposicao potencial nao e memorizacao",
            "snapshot compartilhado NAO faz o risco sumir no contraste M0 x MR: representacoes diferentes podem "
            "aproveitar os mesmos loci de formas diferentes",
            "a assimetria caso x controle e um mecanismo, nao o unico",
            "sobreposicao de locus nao e inevitavel: e escolha de custo, medida no G2",
        ],
    }
    for study in STUDIES:
        rows = members[members["study_id"] == study]
        if rows.empty:
            continue
        report["por_estudo_e_papel"][study] = {
            role: {
                "cluster": describe(rows.loc[rows["member_role"] == role, "n_treino"]),
                "janela": describe(rows.loc[rows["member_role"] == role, "n_janela"]),
            }
            for role in sorted(set(rows["member_role"]))
        }
    for study in STUDIES:
        if (members["study_id"] == study).any():
            report["pareado_por_cluster"][study] = paired_comparison(members, study, column="n_treino")
            report["pareado_por_janela"][study] = paired_comparison(members, study, column="n_janela")
    report["destaque"] = {
        "estudo": STUDY_CLINICAL,
        "por_janela": report["pareado_por_janela"].get(STUDY_CLINICAL, {}),
    }
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--snapshot", required=True, type=Path, help="core_head_snapshot.parquet do G2")
    parser.add_argument("--brazil-variants", required=True, type=Path, help="saida do G1")
    parser.add_argument("--radius-bp", type=int, default=2048,
                        help="metade da janela de extracao (4.096 bp): variantes de treino ate esta distancia "
                             "cabem na janela do membro. Use 4096 para 'janelas se sobrepoem em algum ponto'.")
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args(argv)

    snapshot = pd.read_parquet(args.snapshot.expanduser(),
                               columns=["variant_id", "role", "binary_label", "overlap_cluster_id",
                                        "chrom", "pos_1based"])
    members = pd.read_parquet(args.brazil_variants.expanduser(),
                              columns=["variant_id", "study_id", "member_role", "matched_variant_id",
                                       "binary_label", "primary_panel", "overlap_cluster_id",
                                       "chrom", "pos_1based"])
    if not (snapshot["role"] == ROLE_TRAIN).any():
        print("FALHOU: o snapshot nao tem linhas de treino.")
        return 2

    joined = window_exposure(join_exposure(members, train_exposure_by_cluster(snapshot)), snapshot,
                             radius_bp=args.radius_bp)
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
    }, radius_bp=args.radius_bp)
    report["saidas"] = {"tabela": str(table_path), "relatorio": str(report_path)}
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

    clinical = report["pareado_por_janela"].get(STUDY_CLINICAL, {})
    print(json.dumps({
        "treino": report["treino"],
        "clinico_por_janela": {k: clinical.get(k) for k in ("pares", "caso", "controle",
                                                            "diferenca_caso_menos_controle")},
        "clinico_por_cluster_resumo": report["pareado_por_cluster"].get(STUDY_CLINICAL, {}).get(
            "diferenca_caso_menos_controle"),
        "saidas": report["saidas"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
