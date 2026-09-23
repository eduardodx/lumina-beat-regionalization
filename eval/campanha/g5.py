"""Regras do G5, aplicadas SO a M0 e declaradas antes de qualquer score. Sem torch.

Politica (plano, secao 4.2, fixada em 17/09): a mais isolada cuja macro-AUROC no conjunto de selecao fique a ate
0,01 da melhor, pela media de 3 sementes; empate resolve a favor da mais isolada. Os 0,01 sao regra operacional
declarada, nao prova de equivalencia.

Extracao (o plano so dizia "criterio declarado antes"; PROPOSTA de 23/09, registrada em
`configs/campanha_r03_desenvolvimento.json` antes de qualquer score): aplica-se a regra da politica dentro de cada
extracao e comparam-se as duas no que de fato seria usado -- a media na politica escolhida. Ganha a maior; empate
EXATO fica com `cabecas_172` (menos dimensoes).
"""
from __future__ import annotations

from typing import Any

#: Da mais para a menos isolada.
ISOLAMENTO = ("janela4096", "janela2048", "nenhum")
MARGEM = 0.01
DESEMPATE_DA_EXTRACAO = "cabecas_172"


def politica_escolhida(medias: dict[str, float | None], margem: float = MARGEM) -> str:
    """A mais isolada a ate `margem` da melhor. Recusa macro ausente: escolher com macro incompleta mudaria a
    base da comparacao entre politicas."""
    faltando = sorted(set(ISOLAMENTO) - set(medias))
    if faltando:
        raise ValueError(f"politicas sem resultado: {faltando}")
    indefinidas = [p for p, m in medias.items() if m is None]
    if indefinidas:
        raise ValueError(f"politicas sem macro definida: {indefinidas}")
    melhor = max(medias[p] for p in ISOLAMENTO)
    for politica in ISOLAMENTO:
        if medias[politica] >= melhor - margem - 1e-12:
            return politica
    raise AssertionError("inalcancavel: a melhor sempre esta dentro da margem")


def escolha(medias: dict[tuple[str, str], float | None], *, margem: float = MARGEM,
            desempate: str = DESEMPATE_DA_EXTRACAO) -> dict[str, Any]:
    """`medias[(extracao, politica)]` = media da macro nas sementes. Devolve extracao, politica e o caminho."""
    extracoes = sorted({extracao for extracao, _ in medias})
    por_extracao = {}
    for extracao in extracoes:
        politica = politica_escolhida({p: m for (e, p), m in medias.items() if e == extracao}, margem)
        por_extracao[extracao] = {"politica": politica, "macro_media": medias[(extracao, politica)]}
    maior = max(v["macro_media"] for v in por_extracao.values())
    empatadas = sorted(e for e, v in por_extracao.items() if v["macro_media"] == maior)
    extracao = desempate if len(empatadas) > 1 and desempate in empatadas else empatadas[0]
    return {"extracao": extracao, "politica": por_extracao[extracao]["politica"],
            "macro_media": por_extracao[extracao]["macro_media"], "por_extracao": por_extracao,
            "empate_exato_entre_extracoes": len(empatadas) > 1}
