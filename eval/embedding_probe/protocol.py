"""Protocolo de avaliacao do Mosaic, implementado para o harness de ablacao de features.

Espelha o contrato do consumidor (``PROTOCOLO.md`` §"Core e transferencia entre genes" e
``docs/GUIA_OPERACIONAL_DE_SPLITS.md`` §2.1 do repo lumina-mosaic). Reimplementado aqui em vez de
importado porque o pacote ``mosaic`` puxa ``pysam`` no topo e nao e dependencia deste repo; os
testes travam as regras.

AS REGRAS, LITERALMENTE
-----------------------
Para cada ``run_id = i`` em 0..4::

    teste      = fold i,            SOMENTE gold
    validation = fold (i+1) mod 5,  SOMENTE gold
    treino     = os outros tres folds, gold + consensus

"O consenso dos folds de teste e validation nao entra no treino daquela execucao" -- garantido por
construcao, ja que o treino so olha os tres folds restantes. Nenhuma unidade de bloqueio
(``overlap_cluster_id`` em core_locus, ``gene_transfer_group_id`` em gene_transfer) cruza papeis:
isso e uma ASSERCAO, nao uma esperanca, porque o release ja atribui folds na unidade.

Selecao de hiperparametro: **macro AUROC NAO PONDERADA** de missense/splice/noncoding na
validation. O teste nunca participa dessa escolha.

Agregacao entre folds: **AUROC ponderada por n_P x n_B** de cada fold (README do Mosaic).

DESVIO CONSCIENTE NA FASE DE ABLACAO
------------------------------------
``train_tiers`` permite treinar so com gold. Nao e o protocolo -- e uma reducao de custo para
RANQUEAR configuracoes de features (10.761 gold contra 326.826 exemplos). A configuracao vencedora
deve rodar com ``("gold", "consensus")``, que e o contrato. O harness registra qual foi usado.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

K_FOLDS = 5
GOLD = "gold"
CONSENSUS = "consensus"
DISCRIMINATION_PANELS = ("missense", "splice", "noncoding")

TRACKS = {
    "core_locus": ("core_fold", "overlap_cluster_id"),
    "gene_transfer": ("gene_transfer_fold", "gene_transfer_group_id"),
}


class Split(NamedTuple):
    """Indices (posicoes nas linhas de entrada) de cada papel de uma execucao."""

    run_id: int
    train: list[int]
    validation: list[int]
    test: list[int]

    @property
    def sizes(self) -> dict[str, int]:
        return {"train": len(self.train), "validation": len(self.validation), "test": len(self.test)}


def fold_schedule(run_id: int) -> tuple[int, int, list[int]]:
    """``(fold de teste, fold de validation, folds de treino)`` -- a agenda fixa do protocolo."""
    if run_id not in range(K_FOLDS):
        raise ValueError(f"run_id deve estar em 0..{K_FOLDS - 1}, veio {run_id}")
    test = run_id
    validation = (run_id + 1) % K_FOLDS
    train = [f for f in range(K_FOLDS) if f not in (test, validation)]
    return test, validation, train


def cross_fitted_split(
    folds: Sequence[int | None],
    tiers: Sequence[str],
    *,
    run_id: int,
    train_tiers: tuple[str, ...] = (GOLD, CONSENSUS),
) -> Split:
    """Monta os tres papeis de uma execucao. ``folds[i] is None`` => a variante fica de fora."""
    if len(folds) != len(tiers):
        raise ValueError(f"folds e tiers com tamanhos diferentes: {len(folds)} != {len(tiers)}")
    test_fold, val_fold, train_folds = fold_schedule(run_id)
    train_set, train_ok = set(train_folds), set(train_tiers)
    train, validation, test = [], [], []
    for i, (fold, tier) in enumerate(zip(folds, tiers)):
        if fold is None:
            continue
        fold = int(fold)
        if fold == test_fold and tier == GOLD:
            test.append(i)
        elif fold == val_fold and tier == GOLD:
            validation.append(i)
        elif fold in train_set and tier in train_ok:
            train.append(i)
    return Split(run_id=run_id, train=train, validation=validation, test=test)


def assert_no_unit_leak(split: Split, units: Sequence[str]) -> None:
    """Nenhuma unidade de bloqueio cruza papeis. E a assercao do proprio guia do Mosaic.

    Se isso falhar, o resultado esta contaminado por vazamento e nao vale nada -- entao falha alto.
    """
    roles = {"train": split.train, "validation": split.validation, "test": split.test}
    sets = {name: {units[i] for i in idx} for name, idx in roles.items()}
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        shared = sets[a] & sets[b]
        if shared:
            raise AssertionError(
                f"run {split.run_id}: {len(shared)} unidade(s) de bloqueio em {a} E {b} "
                f"(ex.: {sorted(shared)[:3]}) -- vazamento entre papeis"
            )


def macro_auroc(per_panel: dict[str, float | None]) -> float | None:
    """Macro NAO PONDERADA dos paineis de discriminacao. E a metrica de SELECAO do protocolo.

    Retorna ``None`` se algum painel de discriminacao nao pode ser avaliado -- e melhor nao
    selecionar do que selecionar por uma macro incompleta que muda de base entre configuracoes.
    """
    values = [per_panel.get(p) for p in DISCRIMINATION_PANELS]
    if any(v is None for v in values):
        return None
    return sum(values) / len(values)  # type: ignore[arg-type]


def weighted_cross_fold(
    per_fold: Sequence[tuple[float, int, int]],
) -> float | None:
    """Agrega AUROC entre folds ponderando por ``n_P x n_B`` (regra do README do Mosaic).

    ``per_fold`` = sequencia de ``(auroc, n_pos, n_neg)``. Folds sem as duas classes sao ignorados
    (AUROC indefinida), como manda o release para as coortes brasileiras.
    """
    usable = [(a, p * n) for a, p, n in per_fold if p > 0 and n > 0 and a == a]
    if not usable:
        return None
    total = sum(w for _, w in usable)
    return sum(a * w for a, w in usable) / total if total else None


def worst_panel(per_panel: dict[str, float | None]) -> float | None:
    """Pior AUROC entre os paineis de discriminacao -- um dos quatro campos do resumo do Mosaic."""
    values = [v for p in DISCRIMINATION_PANELS if (v := per_panel.get(p)) is not None]
    return min(values) if len(values) == len(DISCRIMINATION_PANELS) else None
