#!/usr/bin/env python3
"""Confere as cabecas salvas pelo comparador ANTES do congelamento (G6): arquivos, identidades, calibradores e a
reproducao das predicoes ao recarregar.

O comparador grava cada cabeca inteira (pesos, padronizacao, Platt, limiar) para que os estudos do G7 sejam
pontuados pelo sistema congelado, sem retreinar. Isso so vale se o arquivo recarregado der as MESMAS predicoes que
o comparador mediu. Para cada `cabeca_{M0,MR}_h{semente}.pt` da pasta da comparacao:

    1. formato, sistema, semente, extracao e politica batem com o relatorio do comparador;
    2. identidades: sha256 da decisao do G5, do snapshot da politica (o registrado na decisao) e da identidade do
       cache do sistema; no MR, a semente do adapter e o adapter declarado congelado para ela;
    3. Platt e limiar iguais aos do relatorio. Platt com a <= 0 REPROVA: inverteria a ordem das probabilidades, e a
       media do ensemble misturaria escalas invertidas;
    4. receita igual a do codigo atual;
    5. recarregada, a cabeca reproduz as probabilidades de `predicoes_selecao.parquet` (tolerancia 1e-6; a
       diferenca maxima sai impressa);
    6. a media das tres probabilidades reproduz as metricas do ensemble no relatorio;
    7. com `--m0-de-referencia <pasta do comparador da a_1>`: as cabecas do M0 sao IDENTICAS as de la (pesos,
       padronizacao, Platt, limiar, epoca). O M0 nao depende do adapter; a composicao final usa as cabecas do M0 do
       comparador da a_1, e os comparadores da a_2 e da a_3 tem de reproduzi-las.

Nao treina nada e nao le o fold 0 nem os estudos. Grava `conferencia_das_cabecas.json`, com o sha256 de cada
arquivo, que o manifesto do G6 usa.

USO (notebook)
    PYTHONPATH="$WORK" python3 scripts/conferir_cabecas_salvas.py \\
        --comparacao ~/artifacts/redesenho/comparacao_dev_a1 \\
        --cache-m0 ~/artifacts/redesenho/g3_cache/M0 --cache-mr ~/artifacts/redesenho/g3_cache/MR_a1 \\
        --decisao-g5 ~/artifacts/redesenho/g5/g5_decisao.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from eval.campanha import metricas  # noqa: E402
from eval.campanha.cache import sha256_do_arquivo  # noqa: E402
from eval.campanha.leitura_do_cache import carregar_cache  # noqa: E402
from eval.campanha.recortes import adapter_congelado, carregar_campanha  # noqa: E402
from scripts.g5_escolher_extracao_e_politica import sha_da_identidade  # noqa: E402

TOLERANCIA_DA_PROBABILIDADE = 1e-6
TOLERANCIA_DA_METRICA = 1e-6
SISTEMAS = ("M0", "MR")


def conferir_metadados(carga: dict[str, Any], *, sistema: str, semente: int, relatorio: dict[str, Any],
                       esperado: dict[str, Any]) -> list[str]:
    """Problemas nos metadados de UMA cabeca contra o relatorio do comparador e as identidades esperadas. Pura."""
    from eval.campanha.cabeca import FORMATO_DA_CABECA, RECEITA

    problemas = []
    iguais = {"formato": FORMATO_DA_CABECA, "sistema": sistema, "semente": semente,
              "extracao": relatorio["extracao"], "politica": relatorio["politica"],
              "decisao_g5_sha256": esperado["decisao_g5_sha256"], "snapshot_sha256": esperado["snapshot_sha256"],
              "cache_identidade_sha256": esperado["cache_identidade_sha256"][sistema]}
    if sistema == "MR":
        iguais["semente_do_adapter"] = relatorio["semente_do_adapter"]
    for campo, valor in iguais.items():
        if carga.get(campo) != valor:
            problemas.append(f"{campo}: {carga.get(campo)!r} != {valor!r}")
    if carga.get("receita") != RECEITA:
        problemas.append("receita diferente da do codigo atual")
    no_relatorio = next((c for c in relatorio["cabecas"][sistema] if c["semente"] == semente), None)
    if no_relatorio is None:
        problemas.append("cabeca ausente do relatorio")
    else:
        for campo in ("platt", "limiar_de_mcc", "epoca"):
            if carga.get(campo) != no_relatorio.get(campo):
                problemas.append(f"{campo} diferente do relatorio: {carga.get(campo)} != {no_relatorio.get(campo)}")
    platt = carga.get("platt") or {}
    if not platt.get("a", 0) > 0:
        problemas.append(f"Platt com a = {platt.get('a')}: inverteria a ordem das probabilidades")
    return problemas


def conferir_ensemble(probabilidades: dict[str, list[np.ndarray]], y: np.ndarray, paineis: np.ndarray,
                      relatorio: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Recalcula as metricas da media das probabilidades e compara com o relatorio. Pura."""
    saida, problemas = {}, []
    for sistema, lista in probabilidades.items():
        resumo = metricas.resumo(np.mean(lista, axis=0), y, paineis)
        registrado = relatorio["media_das_probabilidades"][sistema]
        saida[sistema] = {}
        for chave in ("macro", "auroc", "auprc"):
            novo, antigo = resumo[chave], registrado[chave]
            saida[sistema][chave] = {"relatorio": antigo, "recalculado": novo}
            if (novo is None) != (antigo is None) or (novo is not None and abs(novo - antigo) > TOLERANCIA_DA_METRICA):
                problemas.append(f"ensemble {sistema} {chave}: recalculado {novo} != relatorio {antigo}")
    return saida, problemas


def diferencas_de_cabeca(a: dict[str, Any], b: dict[str, Any]) -> list[str]:
    """O que difere entre duas cabecas salvas: pesos (bit a bit), padronizacao, Platt, limiar e epoca."""
    import torch

    problemas = []
    if set(a["estado"]) != set(b["estado"]):
        problemas.append("conjuntos de pesos diferentes")
    elif any(not torch.equal(a["estado"][k], b["estado"][k]) for k in a["estado"]):
        problemas.append("pesos diferentes")
    for campo in ("media", "desvio"):
        if not np.array_equal(np.asarray(a[campo]), np.asarray(b[campo])):
            problemas.append(f"{campo} diferente")
    for campo in ("platt", "limiar_de_mcc", "epoca", "semente", "extracao", "politica"):
        if a.get(campo) != b.get(campo):
            problemas.append(f"{campo}: {a.get(campo)} != {b.get(campo)}")
    return problemas


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--comparacao", required=True, type=Path, help="--out-dir do comparador")
    parser.add_argument("--cache-m0", required=True, type=Path)
    parser.add_argument("--cache-mr", required=True, type=Path)
    parser.add_argument("--decisao-g5", required=True, type=Path)
    parser.add_argument("--campanha", type=Path, default=Path("configs/campanha_r03_desenvolvimento.json"))
    parser.add_argument("--m0-de-referencia", type=Path,
                        help="pasta de outro comparador (o da a_1): as cabecas do M0 tem de ser identicas")
    parser.add_argument("--out", type=Path, help="padrao: <comparacao>/conferencia_das_cabecas.json")
    args = parser.parse_args(argv)

    from eval.campanha.cabeca import carregar_cabeca_salva, pontuar_salva

    pasta = args.comparacao.expanduser()
    relatorio = json.loads((pasta / "comparacao_m0_mr.json").read_text(encoding="utf-8"))
    decisao_g5 = json.loads(args.decisao_g5.expanduser().read_text(encoding="utf-8"))
    campanha = carregar_campanha(args.campanha)
    caches = {"M0": args.cache_m0.expanduser(), "MR": args.cache_mr.expanduser()}
    problemas: list[str] = []

    esperado = {"decisao_g5_sha256": sha256_do_arquivo(args.decisao_g5.expanduser()),
                "snapshot_sha256": (decisao_g5.get("snapshots_sha256") or {}).get(relatorio["politica"]),
                "cache_identidade_sha256": {s: sha_da_identidade(c) for s, c in caches.items()}}
    carregados = {s: carregar_cache(c, relatorio["extracao"]) for s, c in caches.items()}
    identidade_mr = carregados["MR"]["identidade"]
    if adapter_congelado(campanha, identidade_mr["semente_do_adapter"])["sha256"] != identidade_mr["adapter_sha256"]:
        problemas.append("o adapter do cache MR nao e o declarado congelado para a semente")

    predicoes = pd.read_parquet(pasta / "predicoes_selecao.parquet")
    tabela = carregados["M0"]["tabela"]
    selecao = np.nonzero(tabela["papel"].astype(str).to_numpy() == "selecao")[0]
    if not np.array_equal(tabela["variant_id"].astype(str).to_numpy()[selecao],
                          predicoes["variant_id"].astype(str).to_numpy()):
        problemas.append("predicoes_selecao.parquet nao esta na ordem das linhas de selecao do cache")
    y = tabela["binary_label"].astype(int).to_numpy()[selecao]
    paineis = tabela["primary_panel"].astype(str).to_numpy()[selecao]

    linhas, probabilidades = [], {s: [] for s in SISTEMAS}
    for sistema in SISTEMAS:
        X = carregados[sistema]["matriz"][selecao]
        for semente in relatorio["sementes_da_cabeca"]:
            arquivo = pasta / f"cabeca_{sistema}_h{semente}.pt"
            if not arquivo.exists():
                problemas.append(f"{arquivo.name} ausente")
                continue
            cabeca = carregar_cabeca_salva(arquivo)
            proprios = conferir_metadados(cabeca, sistema=sistema, semente=int(semente), relatorio=relatorio,
                                          esperado=esperado)
            _, prob = pontuar_salva(cabeca, X)
            probabilidades[sistema].append(prob)
            coluna = f"{sistema}_h{semente}"
            diferenca = (float(np.max(np.abs(prob - predicoes[coluna].to_numpy()))) if coluna in predicoes
                         else float("inf"))
            if not diferenca <= TOLERANCIA_DA_PROBABILIDADE:
                proprios.append(f"recarregada nao reproduz {coluna}: diferenca maxima {diferenca:.3e}")
            linhas.append({"arquivo": arquivo.name, "sha256": sha256_do_arquivo(arquivo), "sistema": sistema,
                           "semente": int(semente), "epoca": cabeca.get("epoca"),
                           "platt": {**cabeca["platt"], "inverte_a_ordem": not cabeca["platt"]["a"] > 0},
                           "limiar_de_mcc": cabeca.get("limiar_de_mcc"),
                           "diferenca_maxima_da_probabilidade": diferenca, "problemas": proprios})
            problemas += [f"{arquivo.name}: {p}" for p in proprios]

    ensemble, problemas_do_ensemble = ({}, ["ensemble nao conferido: faltam cabecas"])
    if all(len(v) == len(relatorio["sementes_da_cabeca"]) for v in probabilidades.values()):
        ensemble, problemas_do_ensemble = conferir_ensemble(probabilidades, y, paineis, relatorio)
    problemas += problemas_do_ensemble

    m0_de_referencia = None
    if args.m0_de_referencia:
        referencia = args.m0_de_referencia.expanduser()
        m0_de_referencia = {"pasta": str(referencia), "cabecas": {}}
        for semente in relatorio["sementes_da_cabeca"]:
            nome = f"cabeca_M0_h{semente}.pt"
            if not (referencia / nome).exists() or not (pasta / nome).exists():
                problemas.append(f"{nome}: ausente aqui ou na referencia")
                continue
            diferencas = diferencas_de_cabeca(carregar_cabeca_salva(pasta / nome),
                                              carregar_cabeca_salva(referencia / nome))
            m0_de_referencia["cabecas"][nome] = diferencas or "identica"
            problemas += [f"{nome} difere da referencia: {d}" for d in diferencas]

    saida = {"formato": "conferencia_das_cabecas_v1", "comparacao": str(pasta), "passou": not problemas,
             "extracao": relatorio["extracao"], "politica": relatorio["politica"],
             "semente_do_adapter": relatorio["semente_do_adapter"], "identidades_esperadas": esperado,
             "cabecas": linhas, "ensemble": ensemble, "m0_de_referencia": m0_de_referencia, "problemas": problemas,
             "tolerancias": {"probabilidade": TOLERANCIA_DA_PROBABILIDADE, "metrica": TOLERANCIA_DA_METRICA}}
    destino = (args.out or pasta / "conferencia_das_cabecas.json").expanduser()
    destino.write_text(json.dumps(saida, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    def _n(valor: Any, formato: str = ".4f") -> str:
        return "-" if valor is None else format(valor, formato)

    print(f"[cabecas] {relatorio['extracao']} | {relatorio['politica']} | adapter {relatorio['semente_do_adapter']}")
    print("  cabeca    sha256        epoca   platt a   platt b   limiar   max|dp|   situacao")
    for linha in linhas:
        platt = linha["platt"]
        print(f"  {linha['sistema']} h{linha['semente']:<5} {linha['sha256'][:12]}  {linha['epoca']!s:>5}  "
              f"{_n(platt['a']):>8}  {_n(platt['b']):>8}  {_n((linha['limiar_de_mcc'] or {}).get('limiar')):>7}  "
              f"{_n(linha['diferenca_maxima_da_probabilidade'], '.1e')}   "
              f"{'ok' if not linha['problemas'] else 'PROBLEMA'}")
    for sistema, valores in ensemble.items():
        print(f"  ensemble {sistema}: " + " | ".join(
            f"{k} {_n(v['recalculado'])} (relatorio {_n(v['relatorio'])})" for k, v in valores.items()))
    if m0_de_referencia is not None:
        iguais = sum(1 for v in m0_de_referencia["cabecas"].values() if v == "identica")
        print(f"  M0 contra {m0_de_referencia['pasta']}: {iguais} de {len(relatorio['sementes_da_cabeca'])} "
              f"cabecas identicas")
    if problemas:
        print(f"\nFALHOU: {len(problemas)} problema(s)")
        for problema in problemas:
            print(f"  - {problema}")
        return 2
    print(f"\nPASSOU: {len(linhas)} cabecas recarregadas reproduzem o comparador. Detalhe em {destino}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
