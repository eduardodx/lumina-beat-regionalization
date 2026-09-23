"""Recortes de desenvolvimento da campanha R03: o que pode ser extraido, pontuado e comparado.

Le a declaracao de `configs/campanha_r03_desenvolvimento.json` e a IMPOE em vez de confiar nela: o papel `test`
(fold 0) e os membros dos estudos brasileiros nao entram em nenhuma etapa de desenvolvimento, nem por engano de
argumento. Sem torch: roda no Windows.

Os papeis que entram:
    train       -> ajusta a cabeca (papel `train` do snapshot da politica)
    validation  -> early stopping, Platt e limiar (fold 1 gold)
    selecao     -> escolha de extracao e politica (so M0) e comparacao exploratoria M0 x MR
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

FORMATO_DA_CAMPANHA = "campanha_r03_desenvolvimento_v1"

PAPEL_TREINO = "train"
PAPEL_VALIDACAO = "validation"
PAPEL_TESTE = "test"
PAPEL_SELECAO = "selecao"
#: O que o desenvolvimento pode tocar. `test` (fold 0) fica de fora por construcao.
PAPEIS_DE_DESENVOLVIMENTO = (PAPEL_TREINO, PAPEL_VALIDACAO, PAPEL_SELECAO)

#: Colunas que a extracao e a cabeca precisam de cada variante.
COLUNAS = ("variant_id", "chrom", "pos_1based", "ref", "alt", "binary_label", "primary_panel",
           "overlap_cluster_id", "label_tier")
CROMOSSOMO_RESERVADO = "chr8"


def normalizar_cromossomo(valor: Any) -> str:
    texto = str(valor)
    return texto if texto.startswith("chr") else f"chr{texto}"


def carregar_campanha(caminho: Path) -> dict[str, Any]:
    """Le a declaracao e confere o que o resto do codigo assume dela."""
    campanha = json.loads(Path(caminho).expanduser().read_text(encoding="utf-8"))
    if campanha.get("formato") != FORMATO_DA_CAMPANHA:
        raise ValueError(f"{caminho}: formato {campanha.get('formato')!r}, esperado {FORMATO_DA_CAMPANHA!r}")
    sementes = campanha["sementes"]
    adapter, cabeca = list(sementes["adapter"]), list(sementes["cabeca"])
    if len(set(adapter)) != len(adapter) or len(set(cabeca)) != len(cabeca):
        raise ValueError("sementes repetidas na declaracao")
    combinacoes = sementes["combinacoes"]
    if [c["adapter"] for c in combinacoes] != adapter or [c["cabeca"] for c in combinacoes] != cabeca:
        raise ValueError("as combinacoes nao pareiam adapter e cabeca na ordem declarada")
    proibidos = campanha["recortes"]["proibidos_no_desenvolvimento"]
    if not any(item.get("papel") == PAPEL_TESTE for item in proibidos):
        raise ValueError("a declaracao precisa proibir o papel `test` no desenvolvimento")
    return campanha


def adapter_congelado(campanha: dict[str, Any], semente: int) -> dict[str, Any]:
    """Identidade declarada do adapter de uma semente. Sem registro, nao ha MR para ela."""
    registro = (campanha.get("adapters_congelados") or {}).get(str(semente))
    if not registro:
        raise ValueError(f"nenhum adapter congelado declarado para a semente {semente}")
    return registro


def _faltando(frame: pd.DataFrame, colunas: tuple[str, ...]) -> list[str]:
    return [c for c in colunas if c not in frame.columns]


def tabela_de_extracao(
    snapshot: pd.DataFrame,
    selecao: pd.DataFrame,
    estudos: pd.DataFrame,
    *,
    papeis: tuple[str, ...] = PAPEIS_DE_DESENVOLVIMENTO,
) -> pd.DataFrame:
    """Uniao das variantes a extrair, com o papel de cada uma.

    Recusa -- nunca filtra em silencio -- papel fora do desenvolvimento, variante do fold 0, membro dos estudos
    brasileiros, chr8 reservado e variante em mais de um papel. O G2 e o `verify_campaign_artifacts.py` ja
    garantem a maior parte disso nos artefatos; aqui a garantia e reconferida no ponto em que as features nascem.
    """
    fora = [p for p in papeis if p not in PAPEIS_DE_DESENVOLVIMENTO]
    if fora:
        raise ValueError(f"papeis fora do desenvolvimento: {fora}. O fold 0 so entra depois do congelamento (G6)")
    for nome, frame, extras in (("snapshot", snapshot, ("role",)), ("selecao", selecao, ())):
        faltando = _faltando(frame, COLUNAS + extras)
        if faltando:
            raise ValueError(f"{nome} sem as colunas {faltando}")
    if "variant_id" not in estudos.columns:
        raise ValueError("estudos sem a coluna variant_id")

    partes = []
    do_snapshot = [p for p in papeis if p != PAPEL_SELECAO]
    if do_snapshot:
        linhas = snapshot[snapshot["role"].isin(do_snapshot)]
        parte = linhas[list(COLUNAS)].copy()
        parte["papel"] = linhas["role"].to_numpy()
        partes.append(parte)
    if PAPEL_SELECAO in papeis:
        parte = selecao[list(COLUNAS)].copy()
        parte["papel"] = PAPEL_SELECAO
        partes.append(parte)
    if not partes:
        raise ValueError("nenhum papel pedido")
    tabela = pd.concat(partes, ignore_index=True)

    problemas: list[str] = []
    repetidas = tabela.loc[tabela["variant_id"].duplicated(), "variant_id"]
    if len(repetidas):
        problemas.append(f"{repetidas.nunique()} variantes em mais de um papel, ex.: {list(repetidas.unique()[:3])}")
    do_fold0 = set(snapshot.loc[snapshot["role"] == PAPEL_TESTE, "variant_id"])
    vazou_fold0 = tabela[tabela["variant_id"].isin(do_fold0)]
    if len(vazou_fold0):
        problemas.append(f"{len(vazou_fold0)} variantes do fold 0 (papel test), ex.: "
                         f"{list(vazou_fold0['variant_id'][:3])}")
    membros = tabela[tabela["variant_id"].isin(set(estudos["variant_id"]))]
    if len(membros):
        problemas.append(f"{len(membros)} membros dos estudos brasileiros, ex.: {list(membros['variant_id'][:3])}")
    reservadas = tabela[tabela["chrom"].map(normalizar_cromossomo) == CROMOSSOMO_RESERVADO]
    if len(reservadas):
        problemas.append(f"{len(reservadas)} variantes no {CROMOSSOMO_RESERVADO} reservado")
    if problemas:
        raise ValueError("tabela de extracao recusada: " + "; ".join(problemas))
    return tabela.sort_values(["papel", "variant_id"], kind="mergesort").reset_index(drop=True)


def hash_da_tabela(tabela: pd.DataFrame) -> str:
    """Hash de COMPOSICAO (variante + papel), independente da ordem das linhas."""
    linhas = sorted(f"{v}\t{p}" for v, p in zip(tabela["variant_id"], tabela["papel"]))
    return hashlib.sha256("\n".join(linhas).encode("utf-8")).hexdigest()


def resumo_da_tabela(tabela: pd.DataFrame) -> dict[str, Any]:
    """Tamanho, classes e paineis por papel -- o que se confere antes de gastar GPU."""
    saida: dict[str, Any] = {}
    for papel, linhas in tabela.groupby("papel", sort=True):
        rotulos = linhas["binary_label"].value_counts().to_dict()
        saida[str(papel)] = {
            "variantes": int(len(linhas)),
            "por_rotulo": {str(k): int(v) for k, v in sorted(rotulos.items(), key=lambda kv: str(kv[0]))},
            "por_painel": {str(k): int(v) for k, v in linhas["primary_panel"].value_counts().sort_index().items()},
            "clusters": int(linhas["overlap_cluster_id"].nunique()),
        }
    return saida
