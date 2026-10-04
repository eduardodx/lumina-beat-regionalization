#!/usr/bin/env python3
"""Fase 1b, preparacao: o tamanho do vies regional para E e E+F, sem extrair nem pontuar nada.

Conta, com as funcoes do proprio avaliador (`scripts/evaluate_regional_bias.py` da entrega: `scoring_rows` e
`definition_frames`), as linhas que ele usaria em cada definicao e celula: variantes com `trained_run` (as que tocam
mais de um fold ficam fora, como no avaliador) e peso do desenho maior que zero. Os pesos, os estratos e a execucao
de pontuacao saem do desenho publicado; nada aqui escolhe um subconjunto por score.

Depois, desconta o que os caches do passo 2 ja cobrem e da o complemento a extrair, para a uniao de todas as
definicoes e so para a `main`. A estimativa de GPU usa a taxa medida na conferencia do passo 2 e e so uma ordem de
grandeza: a extracao das externas ainda nao foi feita, e janelas fora do contig ou com N podem mudar a conta.
Tambem confere os pedidos regionais do manifesto (`outputs/candidate-manifest/requests.parquet`) contra essas linhas.

USO (notebook, no .venv do Mosaic, com `data/annotations/bias-cells/` baixado na entrega)
    PYTHONPATH="$PWD" uv run --project ~/mosaic-v1-2026-09-30 --frozen python scripts/fase1b_inventario.py \\
        --entrega ~/mosaic-v1-2026-09-30 --cache ~/artifacts/redesenho/g3_cache/M0 \\
        --cache ~/artifacts/redesenho/g7_cache/M0 --cache ~/artifacts/mosaic_v1/cache_M0_complemento \\
        --out-dir ~/artifacts/mosaic_v1/inventario_1b_<...>

SAIDAS (em --out-dir, que nao pode existir): inventario_1b.json e complemento_main.txt / complemento_todas.txt (ids).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from eval.campanha.cache import sha256_do_arquivo  # noqa: E402
from scripts.inventario_mosaic_v1 import ids_do_cache  # noqa: E402

CELULAS = "data/annotations/bias-cells"
AVALIADOR = "scripts/evaluate_regional_bias.py"
PROTOCOLO = "config/study-protocol.yaml"
PEDIDOS = "outputs/candidate-manifest/requests.parquet"
RELEASE = "artifacts/mosaic-v1-2026-09-30"
#: s/variante medido na conferencia do passo 2 (512 variantes, 04/10). Ordem de grandeza, nao promessa.
SEGUNDOS_POR_VARIANTE = 0.0636


def carregar_avaliador(entrega: Path) -> ModuleType:
    """O `evaluate_regional_bias.py` DA ENTREGA, carregado pelo caminho (o nome `scripts` colide com o deste repo)."""
    caminho = entrega / AVALIADOR
    spec = importlib.util.spec_from_file_location("mosaic_evaluate_regional_bias", caminho)
    if spec is None or spec.loader is None:
        raise ImportError(f"nao consegui carregar {caminho}")
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


def ler_cortes(entrega: Path) -> list[float]:
    import yaml

    protocolo = yaml.safe_load((entrega / PROTOCOLO).read_text(encoding="utf-8"))
    return list(protocolo["metrics"]["precision_gates"]["bias_estimand"]["phylop241_cuts"])


def linhas_do_avaliador(avaliador: ModuleType, celulas: Path, cortes: list[float]) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """As linhas de cada definicao exatamente como o avaliador as monta, e as anotadas (com e sem `trained_run`)."""
    anotadas = avaliador.scoring_rows(celulas / "scoring", celulas / "views" / "4kb" / "examples.parquet", cortes)
    frames, linhas = avaliador.definition_frames(pd.read_parquet(celulas / "cells.parquet"), anotadas)
    return frames, linhas, anotadas


def contar(frames: dict[str, pd.DataFrame], anotadas: pd.DataFrame) -> dict[str, Any]:
    por_definicao = {}
    for nome, frame in frames.items():
        celula = {}
        for nome_da_celula, g in frame.groupby("cell", sort=True):
            celula[str(nome_da_celula)] = {
                "variantes": int(g["variant_id"].nunique()), "peso_total": float(g["weight"].sum()),
                "por_painel": {str(p): int(n) for p, n in g["primary_panel"].value_counts().sort_index().items()}}
        por_definicao[nome] = celula
    return {"anotadas": int(len(anotadas)), "sem_trained_run": int(anotadas["trained_run"].isna().sum()),
            "por_definicao": por_definicao}


def uniao(frames: dict[str, pd.DataFrame], definicoes: list[str] | None = None) -> set[str]:
    escolhidas = frames if definicoes is None else {k: v for k, v in frames.items() if k in definicoes}
    return set().union(*(set(f["variant_id"].astype(str)) for f in escolhidas.values())) if escolhidas else set()


def conferir_pedidos(pedidos: pd.DataFrame, linhas: pd.DataFrame, alvo: set[str]) -> dict[str, Any]:
    """Os pedidos `regional` do manifesto contra as linhas do avaliador: cobertura e execucao igual ao trained_run."""
    regionais = pedidos[pedidos["study"] == "regional"]
    com_run = regionais[regionais["run"].notna()]
    ids_pedidos = set(regionais["variant_id"].astype(str))
    execucao = linhas.set_index(linhas["variant_id"].astype(str))["run"]
    comuns = com_run[com_run["variant_id"].astype(str).isin(execucao.index)]
    divergentes = int((comuns["run"].astype(int).to_numpy()
                       != execucao.reindex(comuns["variant_id"].astype(str)).to_numpy(dtype=int)).sum())
    return {"pedidos_regionais": int(len(regionais)), "variantes_pedidas": len(ids_pedidos),
            "pedidos_sem_run": int(regionais["run"].isna().sum()),
            "alvo_fora_dos_pedidos": len(alvo - ids_pedidos), "pedidos_fora_do_alvo": len(ids_pedidos - alvo),
            "execucao_diferente_do_trained_run": divergentes}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--entrega", required=True, type=Path)
    parser.add_argument("--cache", action="append", default=[], type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    entrega, destino = args.entrega.expanduser(), args.out_dir.expanduser()
    if destino.exists():
        return _falhar([f"{destino} ja existe; o inventario grava sempre numa pasta nova"])
    celulas = entrega / CELULAS
    try:
        avaliador = carregar_avaliador(entrega)
        frames, linhas, anotadas = linhas_do_avaliador(avaliador, celulas, ler_cortes(entrega))
        cobertos = set().union(*(ids_do_cache(p.expanduser()) for p in args.cache)) if args.cache else set()
        pedidos = pd.read_parquet(entrega / PEDIDOS, columns=["variant_id", "study", "run", "role"])
        release = set(pd.read_parquet(entrega / RELEASE / "clinical-variants.parquet", columns=["variant_id"])
                      ["variant_id"].astype(str))
    except (ImportError, ValueError, KeyError, OSError) as exc:
        return _falhar([f"{type(exc).__name__}: {exc}"])

    alvos = {"main": uniao(frames, ["main"]), "todas": uniao(frames)}
    complementos = {nome: sorted(ids - cobertos) for nome, ids in alvos.items()}
    relatorio = {
        "formato": "fase1b_inventario_v1",
        "natureza": "contagem para preparar a Fase 1b; nao extrai, nao pontua e nao escolhe subconjunto por score",
        "entradas": {nome: sha256_do_arquivo(celulas / nome) for nome in
                     ("cells.parquet", "scoring/examples.parquet", "scoring/annotations.parquet",
                      "views/4kb/examples.parquet")},
        "avaliador_sha256": sha256_do_arquivo(entrega / AVALIADOR),
        "linhas": contar(frames, anotadas),
        "alvos": {nome: {"variantes": len(ids), "no_release": len(ids & release), "nos_caches": len(ids & cobertos),
                         "complemento": len(complementos[nome]),
                         "horas_de_gpu_estimadas": round(len(complementos[nome]) * SEGUNDOS_POR_VARIANTE / 3600, 1)}
                  for nome, ids in alvos.items()},
        "pedidos": conferir_pedidos(pedidos, linhas, alvos["todas"]),
        "estimativa": f"{SEGUNDOS_POR_VARIANTE} s/variante, a taxa da conferencia do passo 2 (ordem de grandeza)",
    }
    destino.mkdir(parents=True)
    (destino / "inventario_1b.json").write_text(json.dumps(relatorio, indent=2, ensure_ascii=False), encoding="utf-8")
    for nome, ids in complementos.items():
        (destino / f"complemento_{nome}.txt").write_text("".join(f"{v}\n" for v in ids), encoding="utf-8")
    imprimir(relatorio)
    print(f"\nINVENTARIO 1b: {destino / 'inventario_1b.json'}")
    return 0


def imprimir(r: dict[str, Any]) -> None:
    print(f"anotadas {r['linhas']['anotadas']:,}; sem trained_run {r['linhas']['sem_trained_run']:,}")
    for nome, celulas in r["linhas"]["por_definicao"].items():
        print(f"  {nome}: " + "; ".join(f"{c} {v['variantes']:,} (peso {v['peso_total']:,.0f})" for c, v in celulas.items()))
    for nome, a in r["alvos"].items():
        print(f"alvo {nome}: {a['variantes']:,} variantes, {a['nos_caches']:,} nos caches, complemento "
              f"{a['complemento']:,} (~{a['horas_de_gpu_estimadas']} h de GPU)")
    print(f"pedidos: {r['pedidos']}")


def _falhar(problemas: list[str]) -> int:
    print(f"FALHOU: {len(problemas)} problema(s)")
    for problema in problemas:
        print(f"  - {problema}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
