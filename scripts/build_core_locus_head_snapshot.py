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

from scripts.import_mosaic_brazil_studies import (  # noqa: E402
    counts_by,
    logical_hash,
    missing_columns,
    sha256_file,
)

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
    "br_lab_any",
)
SNAPSHOT_CONTENT_COLUMNS = (
    "variant_id", "role", "binary_label", "label_tier", "primary_panel", "overlap_cluster_id", "core_fold",
    "chrom", "pos_1based", "ref", "alt",
)

EXCLUSION_STUDY_MEMBERS = "membros_dos_estudos"
EXCLUSION_STUDY_CLUSTERS = "clusters_dos_membros"
EXCLUSION_BROAD_BR = "regra_ampla_brasileira"
EXCLUSION_CHR8 = "chr8_reservado"
EXCLUSION_WINDOW = "janela_dos_membros"
EXCLUSION_SELECTION = "clusters_do_conjunto_de_selecao"

# Politica de exclusao dos VIZINHOS DE CLUSTER dos membros (os membros saem sempre, dos tres recortes).
# Medido em 16/09 no release real: os golds da validacao vivem em 38 clusters e do teste em 31, e o
# br_population_observed e gold (2.640 membros, ~25% do gold do release). Tirar os clusters inteiros dos tres
# recortes zera a validacao -- por isso a politica e explicita, declarada antes de treinar e registrada.
CLUSTER_ALL = "todos"
CLUSTER_TRAIN_ONLY = "treino"
CLUSTER_NONE = "nenhum"
CLUSTER_POLICIES = (CLUSTER_ALL, CLUSTER_TRAIN_ONLY, CLUSTER_NONE)
CLUSTER_POLICY_SCOPE = {
    CLUSTER_ALL: ROLES,
    CLUSTER_TRAIN_ONLY: (ROLE_TRAIN,),
    CLUSTER_NONE: (),
}


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


def within_bp_mask(rows: pd.DataFrame, positions_by_chrom: dict[str, Any], radius_bp: int):
    """True para as linhas que estao a ate `radius_bp` de alguma posicao de referencia, no mesmo cromossomo."""
    import numpy as np

    flags = []
    for chrom, pos in zip(rows["chrom"], rows["pos_1based"]):
        arr = positions_by_chrom.get(str(chrom))
        if arr is None or not len(arr):
            flags.append(False)
            continue
        lo = int(np.searchsorted(arr, pos - radius_bp, side="left"))
        hi = int(np.searchsorted(arr, pos + radius_bp, side="right"))
        flags.append(hi > lo)
    return pd.Series(flags, index=rows.index)


def sorted_positions(frame: pd.DataFrame) -> dict[str, Any]:
    import numpy as np

    return {str(chrom): np.sort(group["pos_1based"].to_numpy(dtype="int64"))
            for chrom, group in frame.groupby("chrom")}


def dropped_counts(rows: pd.DataFrame) -> dict[str, int]:
    labels = rows["binary_label"].astype("Int64")
    return {"n": int(len(rows)), "P": int((labels == 1).sum()), "B": int((labels == 0).sum())}


def apply_exclusions(
    splits: dict[str, pd.DataFrame],
    *,
    study_variants: set[str],
    study_clusters: set[str],
    broad_br: set[str] | None,
    reserve_chr8: bool,
    cluster_policy: str = CLUSTER_ALL,
    member_positions: dict[str, Any] | None = None,
    window_bp: int = 0,
    selection_clusters: set[str] | None = None,
) -> tuple[dict[str, pd.DataFrame], list[dict[str, Any]]]:
    """Aplica as exclusoes em ordem, medindo o custo incremental de cada uma em cada recorte.

    Os membros dos estudos saem SEMPRE dos tres recortes. Os vizinhos de cluster seguem `cluster_policy`, e o
    custo que a politica mais estrita teria e registrado em todo caso (`custo_potencial`), para a escolha ser
    feita com numero e antes de treinar.
    """
    if cluster_policy not in CLUSTER_POLICIES:
        raise ValueError(f"cluster_policy {cluster_policy!r} fora de {CLUSTER_POLICIES}")
    current = {role: rows.copy() for role, rows in splits.items()}
    steps: list[dict[str, Any]] = []

    def step(name: str, keep_mask, *, roles: tuple[str, ...] | None = None, extra: dict | None = None) -> None:
        scope = ROLES if roles is None else roles
        removed: dict[str, dict[str, int]] = {}
        removed_panels: dict[str, dict[str, int]] = {}
        for role, rows in current.items():
            if rows.empty or role not in scope:
                removed[role] = {"n": 0, "P": 0, "B": 0}
                removed_panels[role] = {}
                continue
            mask = keep_mask(rows)
            dropped = rows[~mask]
            removed[role] = dropped_counts(dropped)
            removed_panels[role] = counts_by(dropped, ["primary_panel", "binary_label"])
            current[role] = rows[mask].copy()
        payload = {"exclusao": name, "aplicada_em": list(scope), "removidos": removed,
                   "removidos_por_painel_rotulo": removed_panels,
                   "restantes": {role: role_counts(rows) for role, rows in current.items()},
                   "restantes_por_painel_rotulo": {
                       role: counts_by(rows, ["primary_panel", "binary_label"]) for role, rows in current.items()
                   }}
        steps.append(payload | (extra or {}))

    step(EXCLUSION_STUDY_MEMBERS, lambda rows: ~rows["variant_id"].isin(study_variants))

    # Custo que a politica "todos" teria, medido depois da exclusao dos membros e independente da politica.
    potential = {
        role: dropped_counts(rows[rows["overlap_cluster_id"].isin(study_clusters)])
        | {"clusters_atingidos": int(rows.loc[rows["overlap_cluster_id"].isin(study_clusters),
                                              "overlap_cluster_id"].nunique())}
        for role, rows in current.items()
    }
    step(EXCLUSION_STUDY_CLUSTERS, lambda rows: ~rows["overlap_cluster_id"].isin(study_clusters),
         roles=CLUSTER_POLICY_SCOPE[cluster_policy],
         extra={"politica": cluster_policy, "custo_potencial_se_todos": potential})

    # Alternativa mais barata que o cluster inteiro: tirar do TREINO o que cairia dentro da janela de leitura de
    # algum membro. Zera a exposicao de janela por construcao -- e com ela a assimetria caso x controle.
    if window_bp > 0 and member_positions is not None:
        potential_window = {
            role: dropped_counts(rows[within_bp_mask(rows, member_positions, window_bp)])
            for role, rows in current.items()
        }
        step(EXCLUSION_WINDOW, lambda rows: ~within_bp_mask(rows, member_positions, window_bp),
             roles=(ROLE_TRAIN,),
             extra={"radius_bp": window_bp,
                    "custo_potencial_por_papel": potential_window,
                    "o_que_garante": (
                        f"nenhuma variante de treino a menos de {window_bp} bp de um membro. Isso zera a contagem "
                        f"do medidor com o MESMO raio -- e verificacao de implementacao, nao validacao "
                        f"independente. Para 'nenhuma sobreposicao de sequencia' entre janelas de L bp o raio "
                        f"precisa ser L (4.096), nao L/2: dois centros a 3.000 bp ainda compartilham ~1.096 bp."
                    )})
    else:
        steps.append({"exclusao": EXCLUSION_WINDOW, "status": "nao_aplicada",
                      "motivo": "--window-exclusion-bp 0 (desligado)"})
    if broad_br is not None:
        step(EXCLUSION_BROAD_BR, lambda rows: ~rows["variant_id"].isin(broad_br))
    else:
        steps.append({"exclusao": EXCLUSION_BROAD_BR, "status": "nao_aplicada",
                      "motivo": "lista nao fornecida (--broad-br-variant-ids); snapshot nao pode ser congelado"})
    # O conjunto de selecao comum sai do TREINO de todos os candidatos, por cluster inteiro: nenhum treino pode
    # conter um locus que aparece na selecao. As variantes pontuadas vem do candidato mais restritivo.
    if selection_clusters:
        step(EXCLUSION_SELECTION, lambda rows: ~rows["overlap_cluster_id"].isin(selection_clusters),
             roles=(ROLE_TRAIN,), extra={"clusters": len(selection_clusters)})
    else:
        steps.append({"exclusao": EXCLUSION_SELECTION, "status": "nao_aplicada",
                      "motivo": "--selection-clusters nao fornecido; snapshot ainda NAO e o final de treino"})

    if reserve_chr8:
        step(EXCLUSION_CHR8, lambda rows: rows["chrom"].map(normalize_chrom) != CHR8)
    else:
        steps.append({"exclusao": EXCLUSION_CHR8, "status": "nao_aplicada", "motivo": "--no-reserve-chr8"})
    return current, steps


# ----------------------------------------------------------------------------------------------------- checagens


def check_no_study_leakage(
    splits: dict[str, pd.DataFrame],
    study_variants: set[str],
    study_clusters: set[str],
    *,
    cluster_policy: str = CLUSTER_ALL,
) -> list[str]:
    """Membro de estudo nunca sobra, em recorte nenhum. Vizinho de cluster so e cobrado onde a politica exclui."""
    problems: list[str] = []
    scope = CLUSTER_POLICY_SCOPE[cluster_policy]
    for role, rows in splits.items():
        leaked = sorted(set(rows["variant_id"]) & study_variants)
        if leaked:
            problems.append(f"{role}: {len(leaked)} variantes dos estudos sobraram, ex.: {leaked[:3]}")
        if role in scope:
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
    """Hash de COMPOSICAO (variant_id + papel). Nao muda se um rotulo mudar: para isso ha o hash de conteudo."""
    payload = "\n".join(sorted(
        f"{vid}\t{role}" for vid, role in zip(snapshot["variant_id"], snapshot["role"])
    ))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def snapshot_content_hash(snapshot: pd.DataFrame) -> str:
    return logical_hash(snapshot, SNAPSHOT_CONTENT_COLUMNS)


def load_broad_br(path: Path) -> set[str]:
    if path.suffix == ".parquet":
        frame = pd.read_parquet(path, columns=["variant_id"])
        return set(frame["variant_id"].astype(str))
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def broad_manifest_path(list_path: Path) -> Path:
    return list_path.with_suffix(list_path.suffix + ".manifest.json")


def validate_broad_list(
    broad: set[str],
    frame: pd.DataFrame,
    *,
    manifest: dict[str, Any] | None,
    pb_examples_sha256: str,
    list_sha256: str | None = None,
) -> list[str]:
    """A lista so libera o snapshot se for valida: nao vazia, superconjunto do br_lab_any e do MESMO release.

    O `br_lab_any` publicado passa pelo filtro P/B do Mosaic, que e um subconjunto da regra ampla. Se alguma
    variante marcada no release ficar de fora da lista, a lista esta errada, truncada ou e de outro release --
    e presenca de arquivo nao pode valer como autorizacao para treinar.
    """
    problems: list[str] = []
    if not broad:
        problems.append("lista da regra ampla vazia")
    published = set(frame.loc[frame["br_lab_any"].astype(bool), "variant_id"].astype(str))
    missing = sorted(published - broad)
    if missing:
        problems.append(
            f"lista da regra ampla nao cobre {len(missing)} variantes com br_lab_any no release "
            f"(ex.: {missing[:3]}): lista incompleta ou de outro release"
        )
    if manifest is None:
        problems.append("manifesto da lista ausente (gerado por build_broad_brazilian_variant_list.py)")
    else:
        declared = manifest.get("pb_examples_sha256")
        if declared != pb_examples_sha256:
            problems.append(
                f"manifesto da lista aponta pb_examples sha256 {declared}, mas o release usado tem "
                f"{pb_examples_sha256}"
            )
        if manifest.get("n_variantes") != len(broad):
            problems.append(f"manifesto declara {manifest.get('n_variantes')} variantes, lista tem {len(broad)}")
        # Tamanho e superconjunto nao amarram o CONTEUDO: trocar ids mantendo os dois passaria sem o sha256.
        if list_sha256 is not None and manifest.get("lista_sha256") not in (None, list_sha256):
            problems.append(
                f"manifesto declara lista_sha256 {manifest.get('lista_sha256')}, mas o arquivo tem {list_sha256}"
            )
        elif manifest.get("lista_sha256") is None:
            problems.append("manifesto sem lista_sha256: gere a lista de novo com a versao atual do script")
    return problems


def load_frame(release_root: Path) -> pd.DataFrame:
    examples = pd.read_parquet(release_root / "pb_examples.parquet", columns=list(FRAME_COLUMNS))
    panels = pd.read_parquet(release_root / "pb_panels.parquet", columns=["variant_id", "primary_panel"])
    partitions = pd.read_parquet(
        release_root / "pb_partitions.parquet", columns=["variant_id", "overlap_cluster_id", "core_fold"]
    )
    frame = examples.merge(panels, on="variant_id", how="left", validate="one_to_one")
    return frame.merge(partitions, on="variant_id", how="left", validate="one_to_one")


def load_study_sets(
    release_root: Path, brazil_variants: Path | None
) -> tuple[set[str], set[str], dict[str, Any], list[str]]:
    """As exclusoes vem SEMPRE do membership do release; a saida do G1 e conferencia, nunca substituicao.

    Se o arquivo do G1 estivesse incompleto e virasse a fonte, o G2 excluiria menos e depois checaria vazamento
    contra o mesmo conjunto encolhido -- passaria deixando variantes do estudo no desenvolvimento.
    """
    membership_path = release_root / "studies/brazil/membership.parquet"
    members = pd.read_parquet(membership_path, columns=["variant_id", "overlap_cluster_id"])
    variants = set(members["variant_id"].astype(str))
    clusters = set(members["overlap_cluster_id"].dropna().astype(str))
    origin: dict[str, Any] = {"fonte": str(membership_path), "variantes": len(variants), "clusters": len(clusters)}
    problems: list[str] = []

    if brazil_variants is not None:
        imported = pd.read_parquet(brazil_variants, columns=["variant_id", "overlap_cluster_id"])
        imported_variants = set(imported["variant_id"].astype(str))
        imported_clusters = set(imported["overlap_cluster_id"].dropna().astype(str))
        origin["conferencia_g1"] = {"arquivo": str(brazil_variants), "variantes": len(imported_variants)}
        if imported_variants != variants:
            problems.append(
                f"{brazil_variants}: conjunto de variantes difere do membership do release "
                f"(faltam {len(variants - imported_variants)}, sobram {len(imported_variants - variants)})"
            )
        if imported_clusters != clusters:
            problems.append(
                f"{brazil_variants}: conjunto de overlap_cluster_id difere do membership do release "
                f"(faltam {len(clusters - imported_clusters)}, sobram {len(imported_clusters - clusters)})"
            )
        origin["conferencia_g1"]["bate_com_o_release"] = not problems
    return variants, clusters, origin, problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--release-root", required=True, type=Path)
    parser.add_argument("--brazil-variants", type=Path,
                        help="saida do G1 (brazil_study_variants.parquet); sem ela, le o membership do release")
    parser.add_argument("--broad-br-variant-ids", type=Path,
                        help="variant_id com qualquer SCV de instituicao da lista brasileira (regra ampla)")
    parser.add_argument("--run-id", type=int, default=0)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--cluster-exclusion", choices=CLUSTER_POLICIES, default=CLUSTER_ALL,
                        help="vizinhos de cluster dos membros dos estudos: excluir em todos os recortes "
                             "(padrao), so no treino, ou em nenhum. Os MEMBROS saem sempre dos tres.")
    parser.add_argument("--selection-clusters", type=Path,
                        help="clusters do conjunto de selecao comum: saem do TREINO deste candidato, para que "
                             "nenhum treino contenha um locus que aparece na selecao")
    parser.add_argument("--window-exclusion-bp", type=int, default=0,
                        help="tira do TREINO as variantes a ate N bp de algum membro dos estudos (0 = desligado). "
                             "2048 = metade da janela de 4.096 bp, o que zera a exposicao de janela")
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

    study_variants, study_clusters, study_origin, study_problems = load_study_sets(
        release_root, args.brazil_variants.expanduser() if args.brazil_variants else None
    )

    pb_examples_sha256 = sha256_file(release_root / "pb_examples.parquet")
    broad_br: set[str] | None = None
    broad_manifest: dict[str, Any] | None = None
    broad_problems: list[str] = []
    if args.broad_br_variant_ids:
        list_path = args.broad_br_variant_ids.expanduser()
        broad_br = load_broad_br(list_path)
        manifest_path = broad_manifest_path(list_path)
        if manifest_path.exists():
            broad_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        broad_problems = validate_broad_list(
            broad_br, frame, manifest=broad_manifest, pb_examples_sha256=pb_examples_sha256,
            list_sha256=sha256_file(list_path),
        )

    entry_problems = study_problems + broad_problems
    if entry_problems:
        print(f"FALHOU: {len(entry_problems)} problema(s) nas entradas; nada foi publicado.")
        for problem in entry_problems:
            print(f"  - {problem}")
        return 2

    splits = split_core(frame, run_id=args.run_id, k=args.k)
    before = {role: role_counts(rows) for role, rows in splits.items()}
    selection_clusters = None
    if args.selection_clusters is not None:
        selection_clusters = {line.strip() for line
                              in args.selection_clusters.expanduser().read_text(encoding="utf-8").splitlines()
                              if line.strip()}
    member_positions = sorted_positions(frame[frame["variant_id"].isin(study_variants)])
    splits, steps = apply_exclusions(
        splits, study_variants=study_variants, study_clusters=study_clusters,
        broad_br=broad_br, reserve_chr8=not args.no_reserve_chr8,
        cluster_policy=args.cluster_exclusion,
        member_positions=member_positions, window_bp=args.window_exclusion_bp,
        selection_clusters=selection_clusters,
    )

    if selection_clusters:
        sobrou = sorted(set(splits[ROLE_TRAIN]["overlap_cluster_id"].dropna()) & selection_clusters)
        if sobrou:
            problems_extra = [f"treino: {len(sobrou)} clusters de selecao sobraram, ex.: {sobrou[:3]}"]
        else:
            problems_extra = []
    else:
        problems_extra = []

    problems = problems_extra + check_no_study_leakage(splits, study_variants, study_clusters,
                                                       cluster_policy=args.cluster_exclusion)
    problems += check_clusters_disjoint(splits)
    problems += check_validation_supports_selection(splits)

    out_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = out_dir / "core_head_snapshot.parquet"
    report_path = out_dir / "g2_core_snapshot_report.json"

    # A lista so chega aqui depois de validada (nao vazia, superconjunto do br_lab_any, manifesto do mesmo
    # release): presenca de arquivo nunca libera o congelamento por si.
    pronto = broad_br is not None and not problems
    report: dict[str, Any] = {
        "entradas": {
            "release_root": str(release_root),
            "pb_examples_sha256": pb_examples_sha256,
            "pb_panels_sha256": sha256_file(release_root / "pb_panels.parquet"),
            "pb_partitions_sha256": sha256_file(release_root / "pb_partitions.parquet"),
            "membros_dos_estudos": study_origin,
            "lista_regra_ampla": {
                "arquivo": str(args.broad_br_variant_ids) if args.broad_br_variant_ids else None,
                "sha256": sha256_file(args.broad_br_variant_ids.expanduser()) if args.broad_br_variant_ids else None,
                "variantes": len(broad_br) if broad_br is not None else None,
                "manifesto": broad_manifest,
                "validada": bool(broad_br is not None),
            },
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
        "conjunto_de_selecao": {
            "arquivo": str(args.selection_clusters) if args.selection_clusters else None,
            "clusters": len(selection_clusters) if selection_clusters else 0,
            "aplicado_em": [ROLE_TRAIN] if selection_clusters else [],
            "nota": "sem isto o snapshot NAO e o final de treino: o conjunto de selecao ainda estaria dentro dele",
        },
        "exclusao_por_janela": {
            "radius_bp": args.window_exclusion_bp,
            "aplicada_em": [ROLE_TRAIN] if args.window_exclusion_bp > 0 else [],
            "regra": ("distancia minima entre posicoes" if args.window_exclusion_bp > 0 else "nenhuma"),
            "nota": "raio L/2 (2.048) = nenhuma variante de treino DENTRO da janela do membro; raio L (4.096) = "
                    "nenhuma sobreposicao de sequencia entre as janelas. Sao politicas diferentes, com custos "
                    "diferentes; a escolhida tem de ser declarada. Nao atinge validacao, calibracao nem o treino "
                    "populacional do adapter.",
        },
        "politica_de_cluster": {
            "escolhida": args.cluster_exclusion,
            "recortes_atingidos": list(CLUSTER_POLICY_SCOPE[args.cluster_exclusion]),
            "nota": "os membros dos estudos saem sempre dos tres recortes; esta politica vale so para os "
                    "VIZINHOS de cluster dos membros. Declarar antes de treinar.",
        },
        "antes_das_exclusoes": before,
        "exclusoes": steps,
        "depois_das_exclusoes": {role: role_counts(rows) for role, rows in splits.items()},
        "por_painel_rotulo": {
            role: counts_by(rows, ["primary_panel", "binary_label"]) for role, rows in splits.items()
        },
        "por_tier": {role: counts_by(rows, ["label_tier"]) for role, rows in splits.items()},
        "identidade": {
            "snapshot_id": (f"core_locus_run{args.run_id}_menos_estudos_br_cluster_{args.cluster_exclusion}"
                            f"_janela{args.window_exclusion_bp}"
                            f"{'_menos_selecao' if selection_clusters else ''}"),
            "final_para_treino": bool(selection_clusters),
            "receita_dos_hashes": {
                "composicao": "sha256 das linhas 'variant_id\\trole' ordenadas -- identifica quem esta em cada "
                              "recorte, NAO o conteudo (trocar um rotulo nao muda este hash)",
                "conteudo": f"sha256 das linhas com {list(SNAPSHOT_CONTENT_COLUMNS)} ordenadas",
            },
            "cutoff": "o do release (ClinVar 2026-06)",
            "origem": "derivado do core_locus do release v1, segundo a orientacao do mantenedor em 15/09/2026",
        },
        "pronto_para_congelar": pronto,
        "significado_de_pronto_para_congelar":
            "passou nas checagens da politica ESCOLHIDA e as entradas sao validas. Nao e aprovacao cientifica da "
            "politica de isolamento por locus, que e decidida e declarada fora deste script.",
        "pendencias": [] if broad_br is not None else ["regra ampla brasileira nao aplicada "
                                                       "(--broad-br-variant-ids)"],
        "saidas": {"relatorio": str(report_path)},
    }

    if problems:
        # O relatorio SAI mesmo reprovando: e nele que estao os numeros para decidir o que corrigir. O snapshot
        # e que nao e publicado.
        report["status"] = "FALHOU"
        report["checagens"] = problems
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        print(f"FALHOU: {len(problems)} checagem(ns) do gate G2 nao passaram; o snapshot NAO foi publicado.")
        for problem in problems:
            print(f"  - {problem}")
        print(f"\nDiagnostico completo (custo de cada exclusao, por painel e por papel): {report_path}")
        return 2

    snapshot = snapshot_frame(splits)
    snapshot.to_parquet(snapshot_path, index=False)
    report["status"] = "OK"
    report["checagens"] = "todas passaram"
    report["identidade"] |= {
        "hash_composicao": snapshot_hash(snapshot),
        "hash_conteudo": snapshot_content_hash(snapshot),
        "arquivo_sha256": sha256_file(snapshot_path),
    }
    report["saidas"]["snapshot"] = str(snapshot_path)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps({
        "antes": before,
        "depois": report["depois_das_exclusoes"],
        "politica_de_cluster": args.cluster_exclusion,
        "exclusao_por_janela_bp": args.window_exclusion_bp,
        "hash_composicao": report["identidade"]["hash_composicao"],
        "hash_conteudo": report["identidade"]["hash_conteudo"],
        "pronto_para_congelar": pronto,
        "significado_de_pronto_para_congelar":
            "passou nas checagens da politica ESCOLHIDA e as entradas sao validas. Nao e aprovacao cientifica da "
            "politica de isolamento por locus, que e decidida e declarada fora deste script.",
        "pendencias": report["pendencias"],
        "saidas": report["saidas"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
