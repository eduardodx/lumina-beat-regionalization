"""Baselines diagnosticas do G7: scores FORA dos sistemas, das anotacoes do release. Sem torch.

Tres scores, todos orientados para patogenicidade (maior = mais patogenico) e todos de ORDENACAO -- nenhum e
probabilidade, entao nenhum tem Brier:

    gnomad_rarity       o comparador OFICIAL do Mosaic (`config/comparators.yaml`, mosaic-comparators/v1):
                        -gnomad_v4_af, com not_found e ac0 recebendo AF 0 pela regra oficial. O pareamento dos
                        estudos casa `gnomad_af_bin`, entao dentro do par ela so difere dentro da faixa;
    raridade_no_abraom  -abraom_af, com a variante ausente do ABraOM (`abraom_status` not_found) recebendo 0 --
                        regra NOSSA, analoga a oficial. O 0 e imputacao, nao frequencia medida;
    ausencia_no_abraom  1 se a variante nao esta no ABraOM, 0 se esta. So no estudo clinico, como diagnostico da
                        diferenca de composicao; no populacional a presenca DEFINE os grupos (casos presentes,
                        controles ausentes) e nao se relata como discriminacao.

Cobertura do score nao e cobertura de frequencia observada: `frequencia_observada` conta, por estudo e papel,
quantos membros tem a frequencia medida e quantos receberam o valor da regra de imputacao.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from eval.campanha.estudos import ESTUDO_CLINICO, ESTUDO_POPULACIONAL

GNOMAD_RARITY, RARIDADE_NO_ABRAOM, AUSENCIA_NO_ABRAOM = "gnomad_rarity", "raridade_no_abraom", "ausencia_no_abraom"
COLUNAS_DAS_ANOTACOES = ("variant_id", "gnomad_v4_af", "gnomad_status", "abraom_af", "abraom_status", "present_abraom")

ESPECIFICACOES: dict[str, dict[str, Any]] = {
    GNOMAD_RARITY: {
        "nome": GNOMAD_RARITY, "origem": "comparador oficial do Mosaic (mosaic-comparators/v1)",
        "regra": "-gnomad_v4_af; not_found e ac0 recebem AF 0 (regra oficial do comparador)",
        "papel": {ESTUDO_CLINICO: "frequencia global como referencia fora dos sistemas; o pareamento casa "
                                  "gnomad_af_bin, entao casos e controles tem a mesma distribuicao de status",
                  ESTUDO_POPULACIONAL: "frequencia global como referencia fora dos sistemas; o pareamento casa "
                                       "gnomad_af_bin"}},
    RARIDADE_NO_ABRAOM: {
        "nome": RARIDADE_NO_ABRAOM, "origem": "regra da campanha, analoga ao gnomad_rarity",
        "regra": "-abraom_af; ausente do ABraOM (abraom_status not_found) recebe 0 -- imputacao, nao frequencia medida",
        "papel": {ESTUDO_CLINICO: "frequencia no ABraOM como referencia fora dos sistemas; os casos estao mais "
                                  "presentes no ABraOM que os controles",
                  ESTUDO_POPULACIONAL: "so discrimina entre casos (todos presentes); nos controles (todos ausentes) e "
                                       "constante (0) pela regra de imputacao"}},
    AUSENCIA_NO_ABRAOM: {
        "nome": AUSENCIA_NO_ABRAOM, "origem": "regra da campanha, sobre present_abraom da membership",
        "regra": "1 se present_abraom e falso, 0 se verdadeiro",
        "papel": {ESTUDO_CLINICO: "diagnostico da diferenca de composicao: casos 323 x controles 71 presentes (~4,5x)"},
        "nao_aplicavel": {ESTUDO_POPULACIONAL: "a presenca no ABraOM define os grupos do estudo populacional (casos "
                                               "presentes, controles ausentes): constante em cada grupo, nao se relata "
                                               "como discriminacao"}},
}


def _anotacoes_dos_membros(membros: pd.DataFrame, anotacoes: pd.DataFrame) -> pd.DataFrame:
    """Uma linha por variant_id dos membros, com as anotacoes. Recusa membro sem anotacao ou presenca discordante:
    `scripts/conferir_cobertura_das_baselines.py` ja conferiu os dois no release."""
    if not anotacoes["variant_id"].is_unique:
        raise ValueError("anotacoes com variant_id repetido")
    unicos = membros.drop_duplicates("variant_id")[["variant_id", "present_abraom"]]
    juntos = unicos.merge(anotacoes[list(COLUNAS_DAS_ANOTACOES)], on="variant_id", how="left",
                          suffixes=("", "_anotacao"), indicator=True)
    if (juntos["_merge"] != "both").any():
        raise ValueError(f"{int((juntos['_merge'] != 'both').sum())} membros sem linha nas anotacoes")
    da_anotacao = juntos["present_abraom_anotacao"].astype("boolean").fillna(False).astype(bool)
    if (juntos["present_abraom"].astype(bool) != da_anotacao).any():
        raise ValueError("present_abraom da membership diverge das anotacoes")
    return juntos.drop(columns=["_merge", "present_abraom_anotacao"]).set_index(
        juntos["variant_id"].astype(str), drop=False)


def scores_das_baselines(membros: pd.DataFrame, anotacoes: pd.DataFrame,
                         pontuar_gnomad: Callable[[dict[str, Any]], float | None]) -> dict[str, pd.Series]:
    """Os tres scores, indexados por variant_id. `pontuar_gnomad` e o `comparator_score` do Mosaic com a
    especificacao oficial do gnomad_rarity (uma definicao so, a do protocolo)."""
    a = _anotacoes_dos_membros(membros, anotacoes)
    gnomad = pd.Series([pontuar_gnomad(linha) for linha in a.to_dict("records")], index=a.index, dtype=float)
    af = pd.to_numeric(a["abraom_af"], errors="coerce")
    ausente = a["abraom_status"].astype(str) == "not_found"
    abraom = pd.Series(np.where(np.isfinite(af), -af, np.where(ausente, 0.0, np.nan)), index=a.index, dtype=float)
    ausencia = pd.Series(np.where(a["present_abraom"].astype(bool), 0.0, 1.0), index=a.index, dtype=float)
    return {GNOMAD_RARITY: gnomad, RARIDADE_NO_ABRAOM: abraom, AUSENCIA_NO_ABRAOM: ausencia}


def frequencia_observada(membros: pd.DataFrame, anotacoes: pd.DataFrame) -> dict[str, Any]:
    """Por estudo e papel: quantos membros tem a frequencia MEDIDA e quantos receberam o valor da regra. No gnomAD,
    `ac0` e sitio chamado com AC 0 (status proprio no Mosaic, nem raro nem ausente) e `not_found` e sem registro;
    a regra oficial poe os dois em AF 0. No ABraOM, ausente recebe 0 pela regra da campanha. Pura."""
    a = _anotacoes_dos_membros(membros, anotacoes)
    saida: dict[str, Any] = {}
    for (estudo, papel), grupo in membros.groupby(["study_id", "member_role"], sort=True):
        linhas = a.loc[grupo["variant_id"].astype(str)]
        status = linhas["gnomad_status"].astype(str)
        af_abraom = pd.to_numeric(linhas["abraom_af"], errors="coerce")
        saida[f"{estudo}/{papel}"] = {
            "membros": int(len(linhas)),
            "gnomad": {"af_observada": int((status == "present").sum()), "ac0_sitio_chamado": int((status == "ac0").sum()),
                       "not_found_af_imputada": int((status == "not_found").sum()),
                       "outro_status": int((~status.isin(["present", "ac0", "not_found"])).sum())},
            "abraom": {"af_observada": int(np.isfinite(af_abraom).sum()),
                       "ausente_af_imputada": int((~np.isfinite(af_abraom)
                                                    & (linhas["abraom_status"].astype(str) == "not_found")).sum())},
        }
    return saida
