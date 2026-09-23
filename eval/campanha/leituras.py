"""As duas extracoes candidatas do G5, lidas do MESMO forward.

O lote tem layout fixo ``[ref_0, alt_0, ref_1, alt_1, ...]``: ref e alt de cada variante vao juntos, no mesmo
forward. Isso importa porque cuBLAS/cuDNN/Mamba escolhem algoritmo pelo tamanho do lote e a soma em float nao e
associativa: o mesmo par em lotes de tamanhos diferentes muda ~2e-3 (medido na pesquisa, `PROBE_BATCH_SIZE`).
Com o tamanho de lote FIXO -- e o ultimo lote completado com copias -- toda variante e calculada nas mesmas
condicoes, em M0 e em MR. A vizinhanca no lote NAO e exatamente neutra: no smoke de 23/09 a mesma variante mudou
1,9e-6 (M0) e 1,4e-6 (MR) conforme as outras linhas (a pesquisa tinha medido 0). O que protege a comparacao e o
protocolo: mesma tabela, mesma ordem e mesmo tamanho de lote nos dois sistemas.

- ``cabecas_172``: W.Delta das 7 cabecas lineares (68) + delta das 3 MLP (10) + as cabecas na referencia (78) +
  one-hot da substituicao (16). A candidata da pesquisa de extracao (`eval/embedding_probe/rich.py`).
- ``leitura_antiga_1344``: o que a `RegimeAHead` le -- ``site_ref``, ``variant_repr = alt - ref`` no sitio e a
  media da referencia em ``[f-64, f+64)`` (`_extract_paired_variant_features` + `compute_pool_bounds`), tudo no
  trunk POS-norma de 448.

As duas sao lidas de sequencias SEM mascara: e a representacao que a cabeca clinica usa, e nao a que a perda de
MLM do adapter mediu.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from eval.embedding_probe.rich import LINEAR_HEADS, MLP_HEADS, head_readouts, substitution_onehot
from eval.campanha.layout import EXTRACOES, limites_do_contexto, lote_pareado  # noqa: F401


@torch.inference_mode()
def ler_extracoes(modelo: Any, post: Tensor, *, focal: int, reais: int, refs: list[str],
                  alts: list[str]) -> dict[str, Tensor]:
    """As duas extracoes das ``reais`` primeiras variantes do lote. ``post`` e ``[2n, L, d]`` (trunk POS-norma).

    ``refs``/``alts`` aqui sao as BASES da variante (uma letra), para o one-hot da substituicao.
    """
    if post.ndim != 3 or post.shape[0] % 2:
        raise ValueError(f"post precisa ser [2n, L, d] com pares ref/alt, veio {tuple(post.shape)}")
    if len(refs) < reais or len(alts) < reais:
        raise ValueError("faltam bases para as variantes reais do lote")
    ref_h, alt_h = post[0::2][:reais], post[1::2][:reais]          # [n, L, d]
    site_ref, site_alt = ref_h[:, focal], alt_h[:, focal]         # [n, d]

    leituras = head_readouts(modelo, site_ref, site_alt)
    subst = torch.tensor([substitution_onehot(r, a) for r, a in zip(refs[:reais], alts[:reais])],
                         dtype=site_ref.dtype, device=site_ref.device)
    cabecas = torch.cat([leituras["linear"], leituras["mlp"], leituras["ref"], subst], dim=-1)

    inicio, fim = limites_do_contexto(focal, post.shape[1])
    antiga = torch.cat([site_ref, site_alt - site_ref, ref_h[:, inicio:fim].mean(dim=1)], dim=-1)

    saida = {"cabecas_172": cabecas, "leitura_antiga_1344": antiga}
    for nome, tensor in saida.items():
        esperado = EXTRACOES[nome][1]
        if tensor.shape[-1] != esperado:
            raise RuntimeError(f"{nome} saiu com {tensor.shape[-1]} dims, esperado {esperado} -- o layout das "
                               f"cabecas mudou ({LINEAR_HEADS} + {MLP_HEADS})")
    return saida
