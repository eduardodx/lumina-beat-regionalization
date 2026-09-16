#!/usr/bin/env python3
"""G1: importa e valida o membership dos estudos brasileiros do Mosaic, com as coordenadas de pb_examples.

Roda no notebook (pandas + pyarrow; sem GPU). So le o release e so escreve em --out-dir.
Plano: docs/proposta_mosaic_regionalizacao_desenvolvimento.md, secoes 4.1 e 8 (gate G1).

POR QUE EXISTE
--------------
O estudo brasileiro do Mosaic e o conjunto de avaliacao desta campanha e os pares ja vem materializados: o
protocolo manda importar, nunca recomputar o matching. Alem disso o `variant_id` do Mosaic e um hash opaco
(`var:` + 32 hex de sha256 sobre assembly, sequencia, posicao, ref e alt), entao o membership sozinho nao permite
pontuar nada -- as coordenadas tem de vir do `pb_examples.parquet`. Este script faz esse casamento uma vez, valida
o que o Mosaic promete e publica a tabela que o extrator e a avaliacao vao consumir.

COMO
----
1. Le `studies/brazil/membership.parquet`, `pb_examples.parquet`, `pb_panels.parquet` e `pb_partitions.parquet`.
2. Confere o que o proprio Mosaic assume (`src/mosaic/brazil_study.py`, `comparator_eval/cohorts.py:brazil_views`):
   colunas do schema, estudos e papeis validos, `variant_id` unico em cada visao (coorte completo, casos pareados,
   controles), pareamento 1:1 bidirecional entre `case` e `control`, `unmatched_case` sem par, e `stratum` igual a
   `binary_label|primary_panel|gnomad_af_bin` -- o mesmo que `membership_stratum` produz.
3. Confere a coerencia com o resto do release: rotulo e tier iguais aos do `pb_examples`, painel igual ao do
   `pb_panels`, `overlap_cluster_id` e `core_fold` iguais aos do `pb_partitions`.
4. Publica `brazil_study_variants.parquet` (membership + chrom/pos_1based/ref/alt + sequence_eligible) e um
   relatorio JSON com as contagens por estudo, papel, painel e rotulo.

Qualquer checagem que falhe PARA com codigo 2 e nada e publicado: se o membership nao for o que o protocolo
promete, importar em silencio contamina tudo que vem depois.

O QUE NAO PROVA
---------------
- Nao valida o pareamento em si (nao refazemos o matching, por exigencia do protocolo): valida a estrutura dele.
- Os dois estudos respondem perguntas diferentes -- `br_clinical_evidence` mede participacao de instituicao
  brasileira e `br_population_observed` mede presenca no ABraOM. O relatorio conta os dois separados e nunca soma.
- `br_lab_any` e a marcacao do Mosaic, que nao e o mesmo que nacionalidade da instituicao.

USO (notebook)
--------------
    set -o pipefail
    export WORK=~/testeArq/lumina-beat-regionalization
    PYTHONPATH="$WORK" python3 "$WORK"/scripts/import_mosaic_brazil_studies.py \
        --release-root ~/mosaic-v1 \
        --out-dir ~/artifacts/redesenho/g1_brazil_studies | tee ~/g1.log
    echo "exit=$?"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

STUDY_CLINICAL = "br_clinical_evidence"
STUDY_POPULATION = "br_population_observed"
STUDIES = (STUDY_CLINICAL, STUDY_POPULATION)

ROLE_CASE = "case"
ROLE_UNMATCHED = "unmatched_case"
ROLE_CONTROL = "control"
ROLES = (ROLE_CASE, ROLE_UNMATCHED, ROLE_CONTROL)

MEMBERSHIP_COLUMNS = (
    "variant_id", "study_id", "member_role", "matched_variant_id", "stratum", "label_tier",
    "binary_label", "primary_panel", "gnomad_af_bin", "overlap_cluster_id", "core_fold",
    "br_lab_any", "present_abraom",
)
EXAMPLE_COLUMNS = (
    "variant_id", "chrom", "pos_1based", "ref", "alt", "binary_label", "label_tier", "sequence_eligible",
)
PANEL_COLUMNS = ("variant_id", "primary_panel")
PARTITION_COLUMNS = ("variant_id", "overlap_cluster_id", "core_fold")

# Recomendado pelo Mosaic em brazil_study.py:membership_stratum.
AF_BIN_MISSING = "missing"


# --------------------------------------------------------------------------------------------------- utilidades


def stratum_key(binary_label: Any, primary_panel: Any, gnomad_af_bin: Any) -> str:
    """Replica `mosaic.brazil_study.membership_stratum`: rotulo|painel|bin de AF, com 'missing' para bin nulo."""
    af_bin = AF_BIN_MISSING if gnomad_af_bin is None or pd.isna(gnomad_af_bin) or str(gnomad_af_bin) == "" \
        else str(gnomad_af_bin)
    return "|".join([str(int(binary_label)), str(primary_panel), af_bin])


def logical_hash(frame: pd.DataFrame, columns: tuple[str, ...]) -> str:
    """sha256 sobre as colunas, linha a linha, em ordem lexicografica. Receita declarada no relatorio."""
    rows = frame[list(columns)].astype(str)
    payload = "\n".join(sorted("\t".join(row) for row in rows.itertuples(index=False, name=None)))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def missing_columns(frame: pd.DataFrame, required: tuple[str, ...]) -> list[str]:
    return [column for column in required if column not in frame.columns]


# ----------------------------------------------------------------------------------------------------- checagens


def check_domains(membership: pd.DataFrame) -> list[str]:
    """Estudos, papeis e rotulos dentro do dominio do protocolo."""
    problems: list[str] = []
    unknown_studies = sorted(set(membership["study_id"]) - set(STUDIES))
    if unknown_studies:
        problems.append(f"study_id fora do protocolo: {unknown_studies}")
    unknown_roles = sorted(set(membership["member_role"]) - set(ROLES))
    if unknown_roles:
        problems.append(f"member_role fora do protocolo: {unknown_roles}")
    labels = sorted({int(v) for v in membership["binary_label"].dropna().unique()})
    if not set(labels) <= {0, 1}:
        problems.append(f"binary_label fora de (0, 1): {labels}")
    if membership["binary_label"].isna().any():
        problems.append("binary_label nulo no membership")
    return problems


def check_uniqueness(membership: pd.DataFrame) -> list[str]:
    """As mesmas unicidades que `brazil_views` exige antes de montar as coortes."""
    problems: list[str] = []
    for study, rows in membership.groupby("study_id", sort=True):
        views = {
            "coorte_completo": rows[rows["member_role"].isin([ROLE_CASE, ROLE_UNMATCHED])],
            "casos_pareados": rows[rows["member_role"] == ROLE_CASE],
            "controles": rows[rows["member_role"] == ROLE_CONTROL],
        }
        for name, view in views.items():
            duplicated = view["variant_id"][view["variant_id"].duplicated()].tolist()
            if duplicated:
                problems.append(f"{study}/{name}: variant_id repetido ({len(duplicated)}), ex.: {duplicated[:3]}")
    return problems


def check_pairing(membership: pd.DataFrame) -> list[str]:
    """Pareamento 1:1 bidirecional entre `case` e `control`; `unmatched_case` sem par."""
    problems: list[str] = []
    for study, rows in membership.groupby("study_id", sort=True):
        role_of = dict(zip(rows["variant_id"], rows["member_role"]))
        matched_of = {
            vid: (None if pd.isna(mid) else str(mid))
            for vid, mid in zip(rows["variant_id"], rows["matched_variant_id"])
        }
        cases = rows[rows["member_role"] == ROLE_CASE]["variant_id"].tolist()
        controls = rows[rows["member_role"] == ROLE_CONTROL]["variant_id"].tolist()
        unmatched = rows[rows["member_role"] == ROLE_UNMATCHED]["variant_id"].tolist()

        with_pair = [vid for vid in unmatched if matched_of.get(vid)]
        if with_pair:
            problems.append(f"{study}: unmatched_case com matched_variant_id ({len(with_pair)})")

        for label, ids in (("case", cases), ("control", controls)):
            orphans = [vid for vid in ids if not matched_of.get(vid)]
            if orphans:
                problems.append(f"{study}: {label} sem matched_variant_id ({len(orphans)}), ex.: {orphans[:3]}")

        for vid in cases:
            partner = matched_of.get(vid)
            if not partner:
                continue
            if role_of.get(partner) != ROLE_CONTROL:
                problems.append(f"{study}: case {vid} aponta para {partner}, que nao e control")
            elif matched_of.get(partner) != vid:
                problems.append(f"{study}: par nao bidirecional entre case {vid} e control {partner}")

        for role, ids in ((ROLE_CASE, cases), (ROLE_CONTROL, controls)):
            partners = [matched_of[vid] for vid in ids if matched_of.get(vid)]
            repeated = sorted({p for p in partners if partners.count(p) > 1})
            if repeated:
                problems.append(f"{study}: pareamento nao e 1:1 a partir de {role} ({len(repeated)} reusados)")
    return problems


def check_strata(membership: pd.DataFrame) -> list[str]:
    """`stratum` publicado == rotulo|painel|bin de AF, e caso e controle compartilham o estrato."""
    problems: list[str] = []
    recomputed = [
        stratum_key(label, panel, af_bin)
        for label, panel, af_bin in zip(
            membership["binary_label"], membership["primary_panel"], membership["gnomad_af_bin"]
        )
    ]
    mismatch = [
        (vid, published, rebuilt)
        for vid, published, rebuilt in zip(membership["variant_id"], membership["stratum"], recomputed)
        if str(published) != rebuilt
    ]
    if mismatch:
        problems.append(f"stratum publicado != rotulo|painel|bin_af em {len(mismatch)} linhas, ex.: {mismatch[:3]}")

    for study, rows in membership.groupby("study_id", sort=True):
        stratum_of = dict(zip(rows["variant_id"], rows["stratum"]))
        cases = rows[rows["member_role"] == ROLE_CASE]
        divergent = [
            (vid, stratum_of.get(vid), stratum_of.get(str(mid)))
            for vid, mid in zip(cases["variant_id"], cases["matched_variant_id"])
            if not pd.isna(mid) and str(mid) in stratum_of and stratum_of.get(vid) != stratum_of.get(str(mid))
        ]
        if divergent:
            problems.append(f"{study}: {len(divergent)} pares com estratos diferentes, ex.: {divergent[:3]}")
    return problems


def check_against_release(
    membership: pd.DataFrame,
    examples: pd.DataFrame,
    panels: pd.DataFrame,
    partitions: pd.DataFrame,
) -> list[str]:
    """Rotulo, tier, painel, cluster e fold do membership tem de bater com o resto do release."""
    problems: list[str] = []
    example_of = examples.set_index("variant_id")
    panel_of = panels.set_index("variant_id")["primary_panel"]
    partition_of = partitions.set_index("variant_id")

    unknown = sorted(set(membership["variant_id"]) - set(example_of.index))
    if unknown:
        problems.append(f"{len(unknown)} variant_id do membership ausentes do pb_examples, ex.: {unknown[:3]}")
        return problems

    known = membership.drop_duplicates(subset=["variant_id"]).set_index("variant_id")
    for column, source, source_name in (
        ("binary_label", example_of["binary_label"], "pb_examples"),
        ("label_tier", example_of["label_tier"], "pb_examples"),
        ("primary_panel", panel_of, "pb_panels"),
        ("overlap_cluster_id", partition_of["overlap_cluster_id"], "pb_partitions"),
        ("core_fold", partition_of["core_fold"], "pb_partitions"),
    ):
        if column not in known.columns:
            continue
        joined = known[[column]].join(source.rename("release"), how="left")
        divergent = joined[joined[column].astype(str) != joined["release"].astype(str)]
        if len(divergent):
            sample = divergent.head(3).reset_index().to_dict(orient="records")
            problems.append(f"{column} diverge de {source_name} em {len(divergent)} variantes, ex.: {sample}")
    return problems


# ------------------------------------------------------------------------------------------------------ relatorio


def counts_by(frame: pd.DataFrame, columns: list[str]) -> dict[str, int]:
    if frame.empty:
        return {}
    grouped = frame.groupby(columns, dropna=False).size()
    return {"/".join(str(part) for part in (key if isinstance(key, tuple) else (key,))): int(value)
            for key, value in sorted(grouped.items(), key=lambda item: str(item[0]))}


def build_report(membership: pd.DataFrame, joined: pd.DataFrame, paths: dict[str, str]) -> dict[str, Any]:
    per_study: dict[str, Any] = {}
    for study in STUDIES:
        rows = membership[membership["study_id"] == study]
        if rows.empty:
            continue
        cases = rows[rows["member_role"].isin([ROLE_CASE, ROLE_UNMATCHED])]
        per_study[study] = {
            "linhas": int(len(rows)),
            "por_papel": counts_by(rows, ["member_role"]),
            "coorte_completo_por_rotulo": counts_by(cases, ["binary_label"]),
            "coorte_completo_por_painel_rotulo": counts_by(cases, ["primary_panel", "binary_label"]),
            "controles_por_painel_rotulo": counts_by(rows[rows["member_role"] == ROLE_CONTROL],
                                                    ["primary_panel", "binary_label"]),
            "por_tier": counts_by(rows, ["label_tier"]),
            "br_lab_any": counts_by(rows, ["member_role", "br_lab_any"]),
            "present_abraom": counts_by(rows, ["member_role", "present_abraom"]),
            "pares": int((rows["member_role"] == ROLE_CASE).sum()),
            "casos_sem_par": int((rows["member_role"] == ROLE_UNMATCHED).sum()),
            "clusters_distintos": int(rows["overlap_cluster_id"].nunique()),
        }

    in_both = sorted(
        set(membership[membership["study_id"] == STUDY_CLINICAL]["variant_id"])
        & set(membership[membership["study_id"] == STUDY_POPULATION]["variant_id"])
    )
    pairs_sharing_cluster = 0
    for _, rows in membership.groupby("study_id", sort=True):
        cluster_of = dict(zip(rows["variant_id"], rows["overlap_cluster_id"]))
        cases = rows[rows["member_role"] == ROLE_CASE]
        pairs_sharing_cluster += sum(
            1 for vid, mid in zip(cases["variant_id"], cases["matched_variant_id"])
            if not pd.isna(mid) and cluster_of.get(vid) == cluster_of.get(str(mid))
        )

    return {
        "entradas": paths,
        "membership": {
            "linhas": int(len(membership)),
            "variantes_distintas": int(membership["variant_id"].nunique()),
            "hash_logico": logical_hash(membership, ("variant_id", "study_id", "member_role")),
            "receita_do_hash": "sha256 das linhas 'variant_id\\tstudy_id\\tmember_role' ordenadas",
        },
        "por_estudo": per_study,
        "variantes_nos_dois_estudos": {"n": len(in_both), "exemplos": in_both[:5]},
        "pares_no_mesmo_overlap_cluster": pairs_sharing_cluster,
        "coordenadas": {
            "com_coordenada": int(joined["pos_1based"].notna().sum()),
            "sem_coordenada": int(joined["pos_1based"].isna().sum()),
            "por_cromossomo": counts_by(joined.drop_duplicates(subset=["variant_id"]), ["chrom"]),
            "sequence_eligible_false": int((~joined["sequence_eligible"].astype(bool)).sum()),
        },
        "checagens": "todas passaram",
        "o_que_nao_prova": [
            "nao revalida o pareamento do Mosaic (proibido recomputar), so a estrutura dele",
            "br_lab_any e marcacao do Mosaic, nao nacionalidade da instituicao",
            "os dois estudos medem coisas diferentes (participacao institucional x presenca no ABraOM)",
        ],
    }


# ----------------------------------------------------------------------------------------------------------- main


def load_release(release_root: Path) -> dict[str, pd.DataFrame]:
    return {
        "membership": pd.read_parquet(release_root / "studies/brazil/membership.parquet"),
        "examples": pd.read_parquet(release_root / "pb_examples.parquet", columns=list(EXAMPLE_COLUMNS)),
        "panels": pd.read_parquet(release_root / "pb_panels.parquet", columns=list(PANEL_COLUMNS)),
        "partitions": pd.read_parquet(release_root / "pb_partitions.parquet", columns=list(PARTITION_COLUMNS)),
    }


def validate(tables: dict[str, pd.DataFrame]) -> list[str]:
    problems: list[str] = []
    for name, required in (
        ("membership", MEMBERSHIP_COLUMNS), ("examples", EXAMPLE_COLUMNS),
        ("panels", PANEL_COLUMNS), ("partitions", PARTITION_COLUMNS),
    ):
        absent = missing_columns(tables[name], required)
        if absent:
            problems.append(f"{name}: colunas ausentes {absent}")
    if problems:
        return problems

    membership = tables["membership"]
    problems += check_domains(membership)
    problems += check_uniqueness(membership)
    problems += check_pairing(membership)
    problems += check_strata(membership)
    problems += check_against_release(membership, tables["examples"], tables["panels"], tables["partitions"])
    return problems


def join_coordinates(membership: pd.DataFrame, examples: pd.DataFrame) -> pd.DataFrame:
    coordinates = examples[["variant_id", "chrom", "pos_1based", "ref", "alt", "sequence_eligible"]]
    return membership.merge(coordinates, on="variant_id", how="left", validate="many_to_one")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--release-root", required=True, type=Path,
                        help="raiz do release do Mosaic (ex.: ~/mosaic-v1)")
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args(argv)

    release_root = args.release_root.expanduser()
    out_dir = args.out_dir.expanduser()
    tables = load_release(release_root)

    problems = validate(tables)
    if problems:
        print(f"FALHOU: {len(problems)} checagem(ns) do membership nao passaram; nada foi publicado.")
        for problem in problems:
            print(f"  - {problem}")
        return 2

    joined = join_coordinates(tables["membership"], tables["examples"])
    without_coordinates = joined[joined["pos_1based"].isna()]
    if len(without_coordinates):
        print(f"FALHOU: {len(without_coordinates)} membros sem coordenada no pb_examples.")
        return 2

    out_dir.mkdir(parents=True, exist_ok=True)
    variants_path = out_dir / "brazil_study_variants.parquet"
    report_path = out_dir / "g1_brazil_studies_report.json"
    joined.to_parquet(variants_path, index=False)

    report = build_report(tables["membership"], joined, {
        "release_root": str(release_root),
        "membership": str(release_root / "studies/brazil/membership.parquet"),
        "pb_examples": str(release_root / "pb_examples.parquet"),
    })
    report["saidas"] = {"variantes": str(variants_path), "relatorio": str(report_path)}
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps({
        "linhas": report["membership"]["linhas"],
        "hash_logico": report["membership"]["hash_logico"],
        "por_estudo": {study: data["por_papel"] for study, data in report["por_estudo"].items()},
        "variantes_nos_dois_estudos": report["variantes_nos_dois_estudos"]["n"],
        "saidas": report["saidas"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
