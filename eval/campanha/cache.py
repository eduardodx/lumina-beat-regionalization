"""E/S do cache de features: gravacao atomica, validacao na retomada, completude estrita e trava. Sem torch.

Revisao de 23/09, antes da extracao longa: o cache podia declarar sucesso com variante faltando, um fragmento
truncado por queda da maquina impedia a retomada, o proximo fragmento era numerado pela CONTAGEM de arquivos (uma
lacuna sobrescreveria outro) e a retomada so lia os ids, sem conferir dimensoes, finitude, duplicatas nem a tabela.
"""
from __future__ import annotations

import hashlib
import os
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd

from eval.campanha.layout import EXTRACOES

PADRAO_DO_FRAGMENTO = re.compile(r"^fragmento_(\d{5})\.npz$")
SUFIXO_TEMPORARIO = ".tmp"
ARQUIVO_DA_TRAVA = "extracao.lock"


def nome_do_fragmento(indice: int) -> str:
    return f"fragmento_{indice:05d}.npz"


def indices_existentes(destino: Path) -> list[int]:
    return sorted(int(m.group(1)) for arquivo in destino.glob("fragmento_*.npz")
                  if (m := PADRAO_DO_FRAGMENTO.match(arquivo.name)))


def proximo_indice(destino: Path) -> int:
    """Maior indice + 1. Contar arquivos sobrescreveria um fragmento se houvesse lacuna na numeracao."""
    indices = indices_existentes(destino)
    return (max(indices) + 1) if indices else 0


def limpar_temporarios(destino: Path) -> list[str]:
    """Remove gravacoes que nao terminaram. Por construcao, um `.tmp` nunca e um fragmento valido."""
    removidos = []
    for arquivo in destino.glob(f"fragmento_*.npz{SUFIXO_TEMPORARIO}"):
        arquivo.unlink()
        removidos.append(arquivo.name)
    return sorted(removidos)


def problemas_das_matrizes(variant_id: np.ndarray, matrizes: dict[str, np.ndarray]) -> list[str]:
    """Nomes, formas e finitude de um lote de leituras -- o que se confere antes de gravar e ao reler."""
    problemas = []
    faltando = sorted(set(EXTRACOES) - set(matrizes))
    if faltando:
        problemas.append(f"sem as extracoes {faltando}")
    for nome, matriz in matrizes.items():
        if nome not in EXTRACOES:
            continue
        esperado = (len(variant_id), EXTRACOES[nome][1])
        if tuple(matriz.shape) != esperado:
            problemas.append(f"{nome} com forma {tuple(matriz.shape)}, esperado {esperado}")
        elif not np.isfinite(matriz).all():
            problemas.append(f"{nome} com {int((~np.isfinite(matriz)).sum())} valores nao finitos")
    return problemas


def gravar_fragmento(destino: Path, indice: int, *, variant_id: np.ndarray, papel: np.ndarray,
                     matrizes: dict[str, np.ndarray]) -> Path:
    """Valida, grava num temporario, forca ao disco e so entao renomeia (troca atomica no mesmo sistema de
    arquivos). Uma queda no meio deixa um `.tmp`, nunca um fragmento truncado com o nome final."""
    problemas = problemas_das_matrizes(variant_id, matrizes)
    if len(papel) != len(variant_id):
        problemas.append(f"{len(papel)} papeis para {len(variant_id)} variantes")
    if problemas:
        raise ValueError(f"fragmento {indice} recusado antes de gravar: " + "; ".join(problemas))
    final = destino / nome_do_fragmento(indice)
    if final.exists():
        raise FileExistsError(f"{final.name} ja existe: gravar por cima apagaria variantes ja extraidas")
    temporario = destino / (final.name + SUFIXO_TEMPORARIO)
    with open(temporario, "wb") as arquivo:
        np.savez(arquivo, variant_id=variant_id.astype(str), papel=papel.astype(str),
                 **{nome: matrizes[nome].astype(np.float32) for nome in EXTRACOES})
        arquivo.flush()
        os.fsync(arquivo.fileno())
    os.replace(temporario, final)
    return final


def ler_fragmentos(destino: Path, tabela: pd.DataFrame) -> tuple[set[str], list[str]]:
    """Ids ja extraidos e os problemas encontrados. Qualquer problema invalida a retomada: e o operador quem
    decide o que fazer com um fragmento ruim, nao o codigo em silencio."""
    papel_da_tabela = dict(zip(tabela["variant_id"].astype(str), tabela["papel"].astype(str)))
    feitas: set[str] = set()
    problemas: list[str] = []
    for indice in indices_existentes(destino):
        arquivo = destino / nome_do_fragmento(indice)
        try:
            with np.load(arquivo, allow_pickle=False) as dados:
                conteudo = {chave: dados[chave] for chave in dados.files}
        except Exception as exc:  # noqa: BLE001 -- truncado, corrompido ou nao-npz
            problemas.append(f"{arquivo.name}: ilegivel ({type(exc).__name__})")
            continue
        if "variant_id" not in conteudo or "papel" not in conteudo:
            problemas.append(f"{arquivo.name}: sem variant_id ou papel")
            continue
        ids, papeis = conteudo["variant_id"].astype(str), conteudo["papel"].astype(str)
        problemas += [f"{arquivo.name}: {p}" for p in problemas_das_matrizes(
            ids, {n: conteudo[n] for n in conteudo if n in EXTRACOES})]
        if len(papeis) != len(ids):
            problemas.append(f"{arquivo.name}: {len(papeis)} papeis para {len(ids)} variantes")
        repetidas = sorted(set(ids[pd.Series(ids).duplicated().to_numpy()]) | (feitas & set(ids)))
        if repetidas:
            problemas.append(f"{arquivo.name}: {len(repetidas)} variantes repetidas, ex.: {repetidas[:3]}")
        estranhas = [v for v in ids if v not in papel_da_tabela]
        if estranhas:
            problemas.append(f"{arquivo.name}: {len(estranhas)} variantes fora da tabela, ex.: {estranhas[:3]}")
        trocadas = [v for v, p in zip(ids, papeis) if v in papel_da_tabela and papel_da_tabela[v] != p]
        if trocadas:
            problemas.append(f"{arquivo.name}: {len(trocadas)} variantes com papel diferente do da tabela")
        feitas.update(ids)
    return feitas, problemas


def estado_do_cache(tabela: pd.DataFrame, feitas: set[str]) -> dict[str, Any]:
    """Completo = TODA variante da tabela no cache. Falha de janela nao desconta: estes artefatos foram auditados
    com zero falha, entao uma falha agora e problema de dado, e M0 e MR nao podem ficar com tabelas diferentes."""
    faltando = sorted(set(tabela["variant_id"].astype(str)) - feitas)
    return {"variantes_na_tabela": int(len(tabela)), "variantes_no_cache": int(len(feitas)),
            "faltando": len(faltando), "exemplos_faltando": faltando[:10], "completo": not faltando}


def _processo_vivo(pid: int) -> bool:
    """Consulta sem efeito colateral. No Windows `os.kill(pid, 0)` NAO consulta: vira TerminateProcess e mata o
    processo -- por isso o ramo com OpenProcess/GetExitCodeProcess."""
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        try:
            codigo = ctypes.c_ulong()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(codigo)):
                return False
            return codigo.value == 259  # STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


@contextmanager
def trava(destino: Path) -> Iterator[None]:
    """Uma extracao por diretorio de cache. Trava de processo morto (queda da maquina) e removida com aviso."""
    destino.mkdir(parents=True, exist_ok=True)
    arquivo = destino / ARQUIVO_DA_TRAVA
    if arquivo.exists():
        try:
            dono = int(arquivo.read_text(encoding="utf-8").strip() or "0")
        except ValueError:
            dono = 0
        if dono and dono != os.getpid() and _processo_vivo(dono):
            raise RuntimeError(f"{destino} ja esta sendo extraido pelo processo {dono}; duas extracoes no mesmo "
                               f"cache se sobrescreveriam")
        print(f"[trava] removida trava velha do processo {dono} (nao esta mais vivo)")
        arquivo.unlink()
    descritor = os.open(arquivo, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        os.write(descritor, str(os.getpid()).encode("utf-8"))
    finally:
        os.close(descritor)
    try:
        yield
    finally:
        if arquivo.exists() and arquivo.read_text(encoding="utf-8").strip() == str(os.getpid()):
            arquivo.unlink()


def sha256_do_arquivo(caminho: Path) -> str:
    digest = hashlib.sha256()
    with open(caminho, "rb") as handle:
        for pedaco in iter(lambda: handle.read(1 << 20), b""):
            digest.update(pedaco)
    return digest.hexdigest()


def identidade_do_codigo(raiz: Path, arquivos: tuple[str, ...],
                         pacotes: dict[str, Path] | None = None) -> dict[str, Any]:
    """sha256 de cada arquivo que determina um numero do cache, e um digest por pacote importado (todos os .py,
    em ordem). Substitui a dependencia de alguem lembrar de subir a versao do extrator."""
    saida: dict[str, Any] = {"arquivos": {nome: sha256_do_arquivo(raiz / nome) for nome in arquivos}}
    for nome, pasta in (pacotes or {}).items():
        digest = hashlib.sha256()
        for arquivo in sorted(pasta.rglob("*.py")):
            digest.update(arquivo.relative_to(pasta).as_posix().encode("utf-8") + b"\0")
            digest.update(sha256_do_arquivo(arquivo).encode("utf-8") + b"\n")
        saida[f"pacote_{nome}"] = {"pasta": str(pasta), "sha256": digest.hexdigest()}
    return saida
