"""Le um cache de features do G3 para a cabeca: confere completude e identidade e alinha as matrizes a tabela.

Sem torch. So LE o cache: nada aqui escreve nos diretorios de extracao.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from eval.campanha.cache import indices_existentes, nome_do_fragmento
from eval.campanha.layout import EXTRACOES
from eval.campanha.recortes import hash_do_conteudo

#: Campos da identidade que PODEM diferir entre M0 e MR de uma mesma comparacao. Todo o resto -- tabela, codigo,
#: ambiente, lote, janela, FASTA, checkpoint -- tem de ser igual, senao a diferenca entre os sistemas nao e so o
#: adapter.
CAMPOS_QUE_DIFEREM_ENTRE_SISTEMAS = ("sistema", "adapter_sha256", "semente_do_adapter", "revisao_do_codigo")


class CacheInvalido(ValueError):
    """O cache nao pode sustentar uma comparacao (incompleto, identidade ou tabela nao conferem)."""


def carregar_cache(pasta: Path, extracao: str) -> dict[str, Any]:
    """Tabela, matriz da `extracao` alinhada a tabela, identidade e manifesto de um cache COMPLETO."""
    if extracao not in EXTRACOES:
        raise ValueError(f"extracao {extracao!r} fora de {sorted(EXTRACOES)}")
    pasta = Path(pasta).expanduser()
    for arquivo in ("identidade.json", "manifesto.json", "tabela.parquet"):
        if not (pasta / arquivo).exists():
            raise CacheInvalido(f"{pasta} sem {arquivo}: a extracao nao terminou")
    identidade = json.loads((pasta / "identidade.json").read_text(encoding="utf-8"))
    manifesto = json.loads((pasta / "manifesto.json").read_text(encoding="utf-8"))
    if not manifesto.get("completo"):
        raise CacheInvalido(f"{pasta}: manifesto diz completo={manifesto.get('completo')!r}")
    tabela = pd.read_parquet(pasta / "tabela.parquet")
    if hash_do_conteudo(tabela) != identidade.get("tabela_sha256_conteudo"):
        raise CacheInvalido(f"{pasta}: tabela.parquet nao confere com a identidade")

    ids, blocos = [], []
    for indice in indices_existentes(pasta):
        with open(pasta / nome_do_fragmento(indice), "rb") as handle, np.load(handle, allow_pickle=False) as dados:
            ids.append(dados["variant_id"].astype(str))
            blocos.append(np.asarray(dados[extracao], dtype=np.float32))
    ids_todos = np.concatenate(ids) if ids else np.array([], dtype=str)
    matriz = np.concatenate(blocos) if blocos else np.zeros((0, EXTRACOES[extracao][1]), dtype=np.float32)
    posicao = pd.Series(np.arange(len(ids_todos)), index=ids_todos)
    if posicao.index.has_duplicates:
        raise CacheInvalido(f"{pasta}: variantes repetidas entre fragmentos")
    faltando = set(tabela["variant_id"].astype(str)) - set(ids_todos)
    if faltando or len(ids_todos) != len(tabela):
        raise CacheInvalido(f"{pasta}: {len(faltando)} variantes da tabela fora dos fragmentos, "
                            f"{len(ids_todos) - len(tabela) + len(faltando)} a mais")
    alinhada = matriz[posicao.loc[tabela["variant_id"].astype(str)].to_numpy()]
    if not np.isfinite(alinhada).all():
        raise CacheInvalido(f"{pasta}: {extracao} com valores nao finitos")
    return {"tabela": tabela.reset_index(drop=True), "matriz": alinhada, "identidade": identidade,
            "manifesto": manifesto, "extracao": extracao, "pasta": str(pasta)}


def diferencas_de_identidade(m0: dict[str, Any], mr: dict[str, Any]) -> list[str]:
    """Campos que diferem entre as identidades de M0 e MR alem dos que DEVEM diferir."""
    chaves = (set(m0) | set(mr)) - set(CAMPOS_QUE_DIFEREM_ENTRE_SISTEMAS)
    return sorted(k for k in chaves if m0.get(k) != mr.get(k))


def conferir_par(m0: dict[str, Any], mr: dict[str, Any]) -> None:
    """M0 e MR so podem diferir pelo adapter. Recusa qualquer outra diferenca."""
    if m0["identidade"].get("sistema") != "M0" or mr["identidade"].get("sistema") != "MR":
        raise CacheInvalido(f"sistemas trocados: {m0['identidade'].get('sistema')} x {mr['identidade'].get('sistema')}")
    diferentes = diferencas_de_identidade(m0["identidade"], mr["identidade"])
    if diferentes:
        raise CacheInvalido(f"M0 e MR diferem alem do adapter, em {diferentes}")


def linhas_da_politica(cache: dict[str, Any], snapshot_da_politica: pd.DataFrame) -> dict[str, np.ndarray]:
    """Indices do cache para cada papel de uma politica: treino = o `train` do snapshot DA POLITICA (subconjunto
    do de `nenhum`); validation e selecao vem do cache. Recusa variante de treino da politica fora do cache."""
    tabela = cache["tabela"]
    ids = tabela["variant_id"].astype(str).to_numpy()
    papel = tabela["papel"].astype(str).to_numpy()
    treino_da_politica = set(snapshot_da_politica.loc[snapshot_da_politica["role"] == "train",
                                                     "variant_id"].astype(str))
    fora = treino_da_politica - set(ids[papel == "train"])
    if fora:
        raise CacheInvalido(f"{len(fora)} variantes de treino da politica nao estao no treino do cache "
                            f"(as politicas deviam ser aninhadas)")
    return {"train": np.nonzero((papel == "train") & np.isin(ids, list(treino_da_politica)))[0],
            "validation": np.nonzero(papel == "validation")[0],
            "selecao": np.nonzero(papel == "selecao")[0]}
