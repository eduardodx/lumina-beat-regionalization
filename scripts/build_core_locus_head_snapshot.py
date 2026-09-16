#!/usr/bin/env python3
"""G2: monta o snapshot de treino da cabeca clinica a partir do `core_locus` do release, com as exclusoes.

Roda no notebook (pandas + pyarrow; sem GPU). So le o release e so escreve em --out-dir.
Plano: docs/proposta_mosaic_regionalizacao_desenvolvimento.md, secoes 4.2 e 8 (gate G2).

POR QUE EXISTE
--------------
O Eduardo definiu em 15/09: "treina core_locus e avalia neles". O `core_locus` e cross-fitted em cinco execucoes
(`docs/GUIA_OPERACIONAL_DE_SPLITS.md` §2 e §3) e os casos e controles dos estudos brasileiros sao variantes do
MESMO release -- inclusive gold, no `br_population_observed`, que cai justamente na validacao e no teste gold. Sem
tirar esses membros e os seus `overlap_cluster_id` dos TRES recortes, `study_membership_used_for_training` e
`study_labels_used_for_training` viram falsos e o estudo brasileiro perde a validade.

PAPEL DE CADA RECORTE (nao inverter)
------------------------------------
    treino      folds != run_id e != (run_id+1)%k, gold + consensus   -> treina a cabeca
    validacao   fold (run_id+1)%k, so gold                            -> extracao, hiperparametros, early stopping,
                                                                         Platt e limiar
    teste core  fold run_id, so gold                                  -> so avalia DEPOIS de congelado
O teste nao seleciona nada. Se orientar qualquer escolha, deixa de ser teste reservado.

COMO
----
1. Monta os tres recortes pela agenda oficial e aplica `sequence_eligible` (o modelo exige janela de sequencia).
2. Aplica, em ordem e medindo o custo de cada uma: (a) membros dos dois estudos brasileiros; (b) todas as
   variantes dos `overlap_cluster_id` desses membros; (c) regra ampla brasileira, se a lista for fornecida;
   (d) chr8, enquanto estiver reservado.
3. Checa o que o gate exige: nenhuma variante e nenhum cluster dos estudos sobrando, papeis disjuntos por cluster,
   e as duas classes presentes em missense, splice e noncoding na validacao.
4. Publica o snapshot, o hash logico (identidade para o manifesto) e o relatorio de custo.

Qualquer checagem que falhe PARA com codigo 2 e nada e publicado.

O QUE NAO PROVA
---------------
- A exclusao pela regra ampla brasileira so acontece se a lista for passada em --broad-br-variant-ids; sem ela o
  relatorio registra `nao_aplicada` e o snapshot NAO esta pronto para congelar.
- Os numeros do guia (196.096 / 2.453 / 1.758 para run_id=0) sao ponto de partida, nao o tamanho final.
- Nao ha nada aqui sobre qualidade de rotulo: o tier vem do release.

USO (notebook)
--------------
    set -o pipefail
    export WORK=~/testeArq/lumina-beat-regionalization
    PYTHONPATH="$WORK" python3 "$WORK"/scripts/build_core_locus_head_snapshot.py \
        --release-root ~/mosaic-v1 \
        --brazil-variants ~/artifacts/redesenho/g1_brazil_studies/brazil_study_variants.parquet \
        --run-id 0 \
        --out-dir ~/artifacts/redesenho/g2_core_snapshot | tee ~/g2.log
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.import_mosaic_brazil_studies import counts_by, missing_columns  # noqa: E402

ROLE_TRAIN = "train"
ROLE_VALIDATION = "validation"
ROLE_TEST = "test"
ROLES = (ROLE_TRAIN, ROLE_VALIDATION, ROLE_TEST)

TIER_GOLD = "gold"
TIER_CONSENSUS = "consensus"
TRAIN_TIERS = (TIER_GOLD, TIER_CONSENSUS)

DISCRIMINATION_PANELS = ("missense", "splice", "noncoding")
CHR8 = "chr8"

FRAME_COLUMNS = (
    "variant_id", "chrom", "pos_1based", "ref", "alt", "binary_label", "label_tier", "sequence_eligible",
)

EXCLUSION_STUDY_MEMBERS = "membros_dos_estudos"
EXCLUSION_STUDY_CLUSTERS = "clusters_dos_membros"
EXCLUSION_BROAD_BR = "regra_ampla_brasileira"
EXCLUSION_CHR8 = "chr8_reservado"


# --------------------------------------------------------------------------------------------------- utilidades


def normalize_chrom(value: Any) -> str:
    text = str(value).strip().lower()
    return text if text.startswith("chr") else f"chr{text}"


def fold_roles(run_id: int, k: int = 5) -> dict[str, Any]:
    """Agenda oficial: teste = run_id, validacao = (run_id+1)%k, treino = o resto."""
    if not 0 <= run_id < k:
        raise ValueError(f"run_id {run_id} fora de [0, {k})")
    test = run_id
    validation = (run_id + 1) % k
    train = tuple(fold for fold in range(k) if fold not in {test, validation})
    return {ROLE_TEST: test, ROLE_VALIDATION: validation, ROLE_TRAIN: train}


def split_core(frame: pd.DataFrame, *, run_id: int, k: int = 5) -> dict[str, pd.DataFrame]:
    """Treino gold+consensus nos folds restantes; validacao e teste so gold. Aplica sequence_eligible."""
    eligible = frame[frame["sequence_eligible"].astype(bool)]
    roles = fold_roles(run_id, k)
    folds = eligible["core_fold"].astype("Int64")
    gold = eligible["label_tier"].astype(str) == TIER_GOLD
    trainable = eligible["label_tier"].astype(str).isin(TRAIN_TIERS)
    return {
        ROLE_TRAIN: eligible[folds.isin(list(roles[ROLE_TRAIN])) & trainable].copy(),
        ROLE_VALIDATION: eligible[(folds == roles[ROLE_VALIDATION]) & gold].copy(),
        ROLE_TEST: eligible[(folds == roles[ROLE_TEST]) & gold].copy(),
    }


def role_counts(rows: pd.DataFrame) -> dict[str, int]:
    labels = rows["binary_label"].astype("Int64")
    return {
        "n": int(len(rows)),
        "P": int((labels == 1).sum()),
        "B": int((labels == 0).sum()),
        "clusters": int(rows["overlap_cluster_id"].nunique()) if len(rows) else 0,
    }


# ----------------------------------------------------------------------------------------------------- exclusoes


def apply_exclusions(
    splits: dict[str, pd.DataFrame],
    *,
    study_variants: set[str],
    study_clusters: set[str],
    broad_br: set[str] | None,
    reserve_chr8: bool,
) -> tuple[dict[str, pd.DataFrame], list[dict[str, Any]]]:
    """Aplica as exclusoes em ordem, medindo o custo incremental de cada uma em cada recorte."""
    current = {role: rows.copy() for role, rows in splits.items()}
    steps: list[dict[str, Any]] = []

    def step(name: str, keep_mask) -> None:
        removed: dict[str, dict[str, int]] = {}
        for role, rows in current.items():
            if rows.empty:
                removed[role] = {"n": 0, "P": 0, "B": 0}
                continue
            mask = keep_mask(rows)
            dropped = rows[~mask]
            removed[role] = {
                "n": int(len(dropped)),
                "P": int((dropped["binary_label"].astype("Int64") == 1).sum()),
                "B": int((dropped["binary_label"].astype("Int64") == 0).sum()),
            }
            current[role] = rows[mask].copy()
        steps.append({"exclusao": name, "removidos": removed,
                      "restantes": {role: role_counts(rows) for role, rows in current.items()}})

    step(EXCLUSION_STUDY_MEMBERS, lambda rows: ~rows["variant_id"].isin(study_variants))
    step(EXCLUSION_STUDY_CLUSTERS, lambda rows: ~rows["overlap_cluster_id"].isin(study_clusters))
    if broad_br is not None:
        step(EXCLUSION_BROAD_BR, lambda rows: ~rows["variant_id"].isin(broad_br))
    else:
        steps.append({"exclusao": EXCLUSION_BROAD_BR, "status": "nao_aplicada",
                      "motivo": "lista nao fornecida (--broad-br-variant-ids); snapshot nao pode ser congelado"})
    if reserve_chr8:
        step(EXCLUSION_CHR8, lambda rows: rows["chrom"].map(normalize_chrom) != CHR8)
    else:
        steps.append({"exclusao": EXCLUSION_CHR8, "status": "nao_aplicada", "motivo": "--no-reserve-chr8"})
    return current, steps


# ----------------------------------------------------------------------------------------------------- checagens


def check_no_study_leakage(
    splits: dict[str, pd.DataFrame], study_variants: set[str], study_clusters: set[str]
) -> list[str]:
    problems: list[str] = []
    for role, rows in splits.items():
        leaked = sorted(set(rows["variant_id"]) & study_variants)
        if leaked:
            problems.append(f"{role}: {len(leaked)} variantes dos estudos sobraram, ex.: {leaked[:3]}")
        clusters = sorted(set(rows["overlap_cluster_id"].dropna()) & study_clusters)
        if clusters:
            problems.append(f"{role}: {len(clusters)} clusters dos estudos sobraram, ex.: {clusters[:3]}")
    return problems


def check_clusters_disjoint(splits: dict[str, pd.DataFrame]) -> list[str]:
    problems: list[str] = []
    clusters = {role: set(rows["overlap_cluster_id"].dropna()) for role, rows in splits.items()}
    for left, right in ((ROLE_TRAIN, ROLE_VALIDATION), (ROLE_TRAIN, ROLE_TEST), (ROLE_VALIDATION, ROLE_TEST)):
        shared = clusters[left] & clusters[right]
        if shared:
            problems.append(f"{left} e {right} compartilham {len(shared)} overlap_cluster_id")
    return problems


def check_validation_supports_selection(splits: dict[str, pd.DataFrame]) -> list[str]:
    """A validacao escolhe extracao e hiperparametros: precisa das duas classes em cada painel de discriminacao."""
    problems: list[str] = []
    validation = splits[ROLE_VALIDATION]
    for panel in DISCRIMINATION_PANELS:
        rows = validation[validation["primary_panel"].astype(str) == panel]
        labels = set(rows["binary_label"].astype("Int64").dropna().tolist())
        if labels != {0, 1}:
            problems.append(
                f"validacao/{panel}: n={len(rows)} com classes {sorted(labels)} -- a selecao planejada nao se "
                f"sustenta (resolver ANTES de treinar, nunca trocando de fold depois de ver resultado)"
            )
    if splits[ROLE_TRAIN].empty:
        problems.append("treino vazio depois das exclusoes")
    else:
        train_labels = set(splits[ROLE_TRAIN]["binary_label"].astype("Int64").dropna().tolist())
        if train_labels != {0, 1}:
            problems.append(f"treino com classes {sorted(train_labels)}")
    return problems


# ------------------------------------------------------------------------------------------------------ snapshot


def snapshot_frame(splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    parts = []
    for role in ROLES:
        rows = splits[role].copy()
        rows["role"] = role
        parts.append(rows)
    columns = ["variant_id", "role", "binary_label", "label_tier", "primary_panel",
               "overlap_cluster_id", "core_fold", "chrom", "pos_1based", "ref", "alt"]
    out = pd.concat(parts, ignore_index=True)
    return out[[c for c in columns if c in out.columns]]


def snapshot_hash(snapshot: pd.DataFrame) -> str:
    payload = "\n".join(sorted(
        f"{vid}\t{role}" for vid, role in zip(snapshot["variant_id"], snapshot["role"])
    ))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_broad_br(path: Path) -> set[str]:
    if path.suffix == ".parquet":
        frame = pd.read_parquet(path, columns=["variant_id"])
        return set(frame["variant_id"].astype(str))
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def load_frame(release_root: Path) -> pd.DataFrame:
    examples = pd.read_parquet(release_root / "pb_examples.parquet", columns=list(FRAME_COLUMNS))
    panels = pd.read_parquet(release_root / "pb_panels.parquet", columns=["variant_id", "primary_panel"])
    partitions = pd.read_parquet(
        release_root / "pb_partitions.parquet", columns=["variant_id", "overlap_cluster_id", "core_fold"]
    )
    frame = examples.merge(panels, on="variant_id", how="left", validate="one_to_one")
    return frame.merge(partitions, on="variant_id", how="left", validate="one_to_one")


def load_study_sets(release_root: Path, brazil_variants: Path | None) -> tuple[set[str], set[str], str]:
    if brazil_variants is not None:
        members = pd.read_parquet(brazil_variants, columns=["variant_id", "overlap_cluster_id"])
        source = str(brazil_variants)
    else:
        members = pd.read_parquet(
            release_root / "studies/brazil/membership.parquet", columns=["variant_id", "overlap_cluster_id"]
        )
        source = str(release_root / "studies/brazil/membership.parquet")
    return (set(members["variant_id"].astype(str)),
            set(members["overlap_cluster_id"].dropna().astype(str)), source)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--release-root", required=True, type=Path)
    parser.add_argument("--brazil-variants", type=Path,
                        help="saida do G1 (brazil_study_variants.parquet); sem ela, le o membership do release")
    parser.add_argument("--broad-br-variant-ids", type=Path,
                        help="variant_id com qualquer SCV de instituicao da lista brasileira (regra ampla)")
    parser.add_argument("--run-id", type=int, default=0)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--no-reserve-chr8", action="store_true", help="nao excluir o chr8 (decisao E do Eduardo)")
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args(argv)

    release_root = args.release_root.expanduser()
    out_dir = args.out_dir.expanduser()

    frame = load_frame(release_root)
    absent = missing_columns(frame, FRAME_COLUMNS + ("primary_panel", "overlap_cluster_id", "core_fold"))
    if absent:
        print(f"FALHOU: colunas ausentes no release: {absent}")
        return 2

    study_variants, study_clusters, study_source = load_study_sets(
        release_root, args.brazil_variants.expanduser() if args.brazil_variants else None
    )
    broad_br = load_broad_br(args.broad_br_variant_ids.expanduser()) if args.broad_br_variant_ids else None

    splits = split_core(frame, run_id=args.run_id, k=args.k)
    before = {role: role_counts(rows) for role, rows in splits.items()}
    splits, steps = apply_exclusions(
        splits, study_variants=study_variants, study_clusters=study_clusters,
        broad_br=broad_br, reserve_chr8=not args.no_reserve_chr8,
    )

    problems = check_no_study_leakage(splits, study_variants, study_clusters)
    problems += check_clusters_disjoint(splits)
    problems += check_validation_supports_selection(splits)
    if problems:
        print(f"FALHOU: {len(problems)} checagem(ns) do gate G2 nao passaram; nada foi publicado.")
        for problem in problems:
            print(f"  - {problem}")
        return 2

    snapshot = snapshot_frame(splits)
    out_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = out_dir / "core_head_snapshot.parquet"
    report_path = out_dir / "g2_core_snapshot_report.json"
    snapshot.to_parquet(snapshot_path, index=False)

    pronto = broad_br is not None
    report: dict[str, Any] = {
        "entradas": {
            "release_root": str(release_root),
            "membros_dos_estudos": study_source,
            "lista_regra_ampla": str(args.broad_br_variant_ids) if args.broad_br_variant_ids else None,
        },
        "agenda": {"run_id": args.run_id, "k": args.k, "folds": {
            role: (list(value) if isinstance(value, tuple) else value)
            for role, value in fold_roles(args.run_id, args.k).items()
        }},
        "papel_de_cada_recorte": {
            ROLE_TRAIN: "treina a cabeca",
            ROLE_VALIDATION: "extracao, hiperparametros, early stopping, Platt e limiar",
            ROLE_TEST: "avalia so depois de congelado; nao seleciona nada",
        },
        "antes_das_exclusoes": before,
        "exclusoes": steps,
        "depois_das_exclusoes": {role: role_counts(rows) for role, rows in splits.items()},
        "por_painel_rotulo": {
            role: counts_by(rows, ["primary_panel", "binary_label"]) for role, rows in splits.items()
        },
        "por_tier": {role: counts_by(rows, ["label_tier"]) for role, rows in splits.items()},
        "identidade": {
            "snapshot_id": f"core_locus_run{args.run_id}_menos_estudos_br",
            "hash_logico": snapshot_hash(snapshot),
            "receita_do_hash": "sha256 das linhas 'variant_id\\trole' ordenadas",
            "cutoff": "o do release (ClinVar 2026-06)",
            "origem": "derivado do core_locus do release v1, segundo a orientacao do mantenedor em 15/09/2026",
        },
        "pronto_para_congelar": pronto,
        "pendencias": [] if pronto else ["regra ampla brasileira nao aplicada (--broad-br-variant-ids)"],
        "checagens": "todas passaram",
        "saidas": {"snapshot": str(snapshot_path), "relatorio": str(report_path)},
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps({
        "antes": before,
        "depois": report["depois_das_exclusoes"],
        "hash_logico": report["identidade"]["hash_logico"],
        "pronto_para_congelar": pronto,
        "pendencias": report["pendencias"],
        "saidas": report["saidas"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
