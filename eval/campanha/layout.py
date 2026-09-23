"""Layout do lote e das extracoes: a parte sem torch, testavel no Windows.

A numerica das leituras depende do TAMANHO do forward (cuBLAS/cuDNN/Mamba escolhem algoritmo por ele; a pesquisa
mediu ~2e-3 entre tamanhos) e, muito menos, das outras linhas do lote (1e-6 no smoke de 23/09). Por isso o lote tem
tamanho fixo, o ultimo e completado com copias e a ordem da tabela e fixa: toda variante e calculada nas mesmas
condicoes, em M0 e em MR.
"""
from __future__ import annotations

RAIO_DO_CONTEXTO = 64
#: Nome -> (blocos na ordem gravada, dimensao total). A ordem faz parte da identidade do cache.
EXTRACOES: dict[str, tuple[tuple[str, ...], int]] = {
    "cabecas_172": (("heads_lin", "heads_mlp", "heads_ref", "subst"), 172),
    "leitura_antiga_1344": (("site_ref", "variant_repr", "local_context"), 1344),
}


def limites_do_contexto(focal: int, comprimento: int, raio: int = RAIO_DO_CONTEXTO) -> tuple[int, int]:
    """[inicio, fim) da media local -- a mesma conta de `compute_pool_bounds` da leitura antiga."""
    return max(0, focal - raio), min(comprimento, focal + raio)


def lote_pareado(refs: list[str], alts: list[str], tamanho: int) -> tuple[list[str], int]:
    """Sequencias no layout ``[ref_0, alt_0, ref_1, alt_1, ...]``, completadas ate ``tamanho`` pares com copias
    do ultimo par. Devolve as sequencias e quantos pares sao REAIS."""
    if not refs or len(refs) != len(alts):
        raise ValueError("refs e alts precisam ter o mesmo tamanho, maior que zero")
    if len(refs) > tamanho:
        raise ValueError(f"{len(refs)} pares nao cabem num lote de {tamanho}")
    reais = len(refs)
    refs = refs + [refs[-1]] * (tamanho - reais)
    alts = alts + [alts[-1]] * (tamanho - reais)
    sequencias: list[str] = []
    for ref, alt in zip(refs, alts):
        sequencias += [ref, alt]
    return sequencias, reais
