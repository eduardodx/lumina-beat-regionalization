#!/usr/bin/env python3
"""G4: separa o plano de janelas do adapter em treino e validacao POR LOCO, sem sobreposicao de sequencia.

Roda no notebook e no Windows (pandas + pyarrow; sem GPU, sem FASTA, sem rede). So le o plano; escreve em --out-dir.
Plano: docs/proposta_mosaic_regionalizacao_desenvolvimento.md, secao 5.1.

POR QUE EXISTE
--------------
A validacao do adapter tem de responder "ele aprendeu estrutura populacional?", nao "ele decorou estes locos".
Separar as janelas ao acaso nao responde isso: duas janelas de 4.096 bp a 500 bp de distancia compartilham 87% da
sequencia, entao a validacao estaria medindo memorizacao do mesmo trecho que o treino viu.

Aqui a unidade de separacao e o LOCO, nao a janela. Janelas cujas janelas se sobrepoem entram no mesmo loco por
encadeamento (ligacao simples), e o loco inteiro vai para um lado so. O encadeamento e necessario, nao conservador
demais: se A cobre B e B cobre C, mandar A para o treino e C para a validacao ainda obrigaria B a sobrepor um dos
dois.

A DISJUNCAO E VERIFICADA, NAO ASSUMIDA. No fim o script varre os dois recortes por cromossomo e falha com codigo 2
se qualquer janela de validacao tocar qualquer janela de treino. Sem essa checagem, um erro de encadeamento
passaria como "separado" e contaminaria a unica medida honesta do adapter.

O QUE ELE PRESERVA
------------------
A mistura 60/40 e os bins de AF sao a receita declarada: separar sem olha-los devolveria uma validacao com outra
composicao, e a comparacao deixaria de valer. A escolha dos locos e gulosa por celula `fonte x bin de AF`, e o
relatorio publica a fracao atingida em cada celula -- se alguma ficar fora da tolerancia, sai como pendencia.

O QUE NAO PROVA
---------------
- Nao garante independencia estatistica: locos distintos ainda podem ser parecidos (parologos, repeticoes,
  familias de genes). Garante ausencia de SOBREPOSICAO DE SEQUENCIA, que e o vazamento grosseiro.
- Nao decide o tamanho da validacao: `--fracao-validacao` e declarado.

USO (notebook)
--------------
    export WORK=~/testeArq/lumina-beat-regionalization
    PYTHONPATH="$WORK" python3 "$WORK"/scripts/split_adapter_plan_by_locus.py \\
        --plano ~/artifacts/redesenho/g4_plano/plano_de_janelas.parquet \\
        --fracao-validacao 0.1 --out-dir ~/artifacts/redesenho/g4_split
    echo "exit=$?"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.import_mosaic_brazil_studies import sha256_file  # noqa: E402

COLUNAS = ("variant_id", "chrom", "pos_1based", "focal_index", "window_start", "fonte", "af_bin")

#: Quanto a fracao de validacao de uma celula pode se afastar da pedida antes de virar pendencia.
TOLERANCIA_DA_CELULA = 0.05


def atribuir_locus(plano: pd.DataFrame, *, window_bp: int, folga_bp: int = 0) -> pd.Series:
    """Id do loco de cada janela, por encadeamento de intervalos que se tocam (ligacao simples).

    Duas janelas ficam no mesmo loco quando `[window_start, window_start + window_bp)` se sobrepoem, ou distam
    menos que `folga_bp`. O id e estavel: depende so das coordenadas, nao da ordem do arquivo nem da semente.
    """
    ordenado = plano.sort_values(["chrom", "window_start", "variant_id"], kind="mergesort")
    ids = pd.Series(index=plano.index, dtype="object")
    for chrom, grupo in ordenado.groupby("chrom", sort=True):
        inicio_atual = -1
        fim_atual = -1
        contador = -1
        for indice, comeco in zip(grupo.index, grupo["window_start"].astype(int)):
            fim = comeco + window_bp
            if comeco >= fim_atual + folga_bp or contador < 0:
                contador += 1
                inicio_atual, fim_atual = comeco, fim
            else:
                fim_atual = max(fim_atual, fim)
            ids[indice] = f"{chrom}:{contador}"
    return ids


def escolher_validacao(
    plano: pd.DataFrame, *, fracao: float, rng: np.random.Generator
) -> set[str]:
    """Locos da validacao, escolhidos guloso por celula `fonte x bin de AF`.

    Escolher ao acaso puro devolveria uma validacao com outra mistura e outro espectro de AF -- e a receita
    declara os dois. O guloso so aceita um loco enquanto alguma celula dele ainda tem deficit.
    """
    alvo = {chave: fracao * len(grupo)
            for chave, grupo in plano.groupby(["fonte", "af_bin"], sort=True)}
    obtido: dict[Any, float] = {chave: 0.0 for chave in alvo}

    locos = sorted(plano["locus_id"].unique())
    ordem = rng.permutation(len(locos))
    escolhidos: set[str] = set()
    por_loco = {loco: grupo for loco, grupo in plano.groupby("locus_id", sort=True)}

    for posicao in ordem:
        loco = locos[int(posicao)]
        grupo = por_loco[loco]
        celulas = grupo.groupby(["fonte", "af_bin"], sort=True).size()
        # Aceita enquanto QUALQUER celula do loco tiver deficit: um loco e indivisivel, entao exigir deficit em
        # todas travaria a escolha em loco misto.
        if not any(obtido.get(chave, 0.0) < alvo.get(chave, 0.0) for chave in celulas.index):
            continue
        escolhidos.add(loco)
        for chave, quantidade in celulas.items():
            obtido[chave] = obtido.get(chave, 0.0) + int(quantidade)
    return escolhidos


def violacoes_de_disjuncao(
    treino: pd.DataFrame, validacao: pd.DataFrame, *, window_bp: int
) -> list[dict[str, Any]]:
    """Pares treino x validacao cujas janelas se tocam. Varredura por cromossomo, exata.

    Ordena os dois recortes juntos e percorre: se duas janelas adjacentes na ordem se sobrepoem e vem de lados
    diferentes, e vazamento. Sem esta varredura, um erro de encadeamento passaria como "separado".
    """
    marcado = pd.concat([
        treino.assign(_lado="treino"), validacao.assign(_lado="validacao")], ignore_index=True)
    problemas: list[dict[str, Any]] = []
    for chrom, grupo in marcado.groupby("chrom", sort=True):
        ordenado = grupo.sort_values("window_start", kind="mergesort")
        starts = ordenado["window_start"].astype(int).to_numpy()
        lados = ordenado["_lado"].to_numpy()
        ids = ordenado["variant_id"].to_numpy()
        # Fim maximo visto ate aqui de cada lado; basta compara-lo com o inicio atual.
        ultimo_fim = {"treino": -1, "validacao": -1}
        ultimo_id = {"treino": None, "validacao": None}
        for comeco, lado, vid in zip(starts, lados, ids):
            outro = "validacao" if lado == "treino" else "treino"
            if comeco < ultimo_fim[outro]:
                problemas.append({"chrom": str(chrom), "variant_id": str(vid), "lado": str(lado),
                                  "toca": str(ultimo_id[outro])})
            ultimo_fim[lado] = max(ultimo_fim[lado], comeco + window_bp)
            ultimo_id[lado] = str(vid)
    return problemas


def composicao(recorte: pd.DataFrame) -> dict[str, Any]:
    if recorte.empty:
        return {"n": 0}
    return {
        "n": int(len(recorte)),
        "locos": int(recorte["locus_id"].nunique()),
        "por_fonte": {str(f): int(q) for f, q in recorte["fonte"].value_counts().items()},
        "fracao_global": round(float((recorte["fonte"] == "global").mean()), 4),
        "por_af_bin": {str(b): int(q) for b, q in recorte["af_bin"].value_counts().sort_index().items()},
        "por_cromossomo": {str(c): int(q) for c, q in recorte["chrom"].value_counts().sort_index().items()},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--plano", required=True, type=Path)
    parser.add_argument("--window-bp", type=int, default=4096)
    parser.add_argument("--folga-bp", type=int, default=0,
                        help="exigir esta distancia ALEM da nao-sobreposicao entre locos")
    parser.add_argument("--fracao-validacao", type=float, default=0.1)
    parser.add_argument("--tolerancia-da-celula", type=float, default=TOLERANCIA_DA_CELULA,
                        help="desvio ABSOLUTO admitido por celula; no alvo 0,10 o padrao 0,05 admite 0,05 a 0,15")
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args(argv)

    if not 0.0 < args.fracao_validacao < 1.0:
        print(f"FALHOU: --fracao-validacao {args.fracao_validacao} fora de (0,1)")
        return 2

    plano = pd.read_parquet(args.plano.expanduser())
    faltando = [c for c in COLUNAS if c not in plano.columns]
    if faltando:
        print(f"FALHOU: faltam colunas {faltando} em {args.plano}")
        return 2

    plano = plano.copy()
    plano["locus_id"] = atribuir_locus(plano, window_bp=args.window_bp, folga_bp=args.folga_bp)
    rng = np.random.default_rng(args.seed)
    locos_validacao = escolher_validacao(plano, fracao=args.fracao_validacao, rng=rng)

    mascara = plano["locus_id"].isin(locos_validacao)
    validacao = plano[mascara].reset_index(drop=True)
    treino = plano[~mascara].reset_index(drop=True)

    problemas = violacoes_de_disjuncao(treino, validacao, window_bp=args.window_bp)

    tamanhos = plano.groupby("locus_id", sort=False).size()
    pendencias: list[str] = []
    por_celula: dict[str, Any] = {}
    for (fonte, af_bin), grupo in plano.groupby(["fonte", "af_bin"], sort=True):
        na_validacao = int(mascara[grupo.index].sum())
        atingida = round(na_validacao / len(grupo), 4)
        por_celula[f"{fonte}|{af_bin}"] = {"total": int(len(grupo)), "validacao": na_validacao,
                                           "fracao": atingida}
        if abs(atingida - args.fracao_validacao) > args.tolerancia_da_celula:
            pendencias.append(f"celula {fonte}|{af_bin} com fracao {atingida}, pedida "
                              f"{args.fracao_validacao} (+/- {args.tolerancia_da_celula})")
    if problemas:
        pendencias.append(f"{len(problemas)} janelas de lados opostos se tocam: a separacao NAO vale")
    if validacao.empty or treino.empty:
        pendencias.append("um dos recortes ficou vazio")

    out_dir = args.out_dir.expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    saidas: dict[str, Any] = {}
    if not pendencias:
        for nome, recorte in (("plano_treino", treino), ("plano_validacao", validacao)):
            caminho = out_dir / f"{nome}.parquet"
            recorte.to_parquet(caminho, index=False)
            saidas[nome] = str(caminho)
            saidas[f"{nome}_sha256"] = sha256_file(caminho)
    if problemas:
        pd.DataFrame(problemas).to_parquet(out_dir / "violacoes.parquet", index=False)

    relatorio: dict[str, Any] = {
        "entrada": {"plano": str(args.plano), "plano_sha256": sha256_file(args.plano.expanduser())},
        "receita": {"window_bp": args.window_bp, "folga_bp": args.folga_bp,
                    "fracao_validacao": args.fracao_validacao, "seed": args.seed,
                    "tolerancia_da_celula": args.tolerancia_da_celula,
                    "leitura_da_tolerancia": (
                        f"ABSOLUTA: no alvo {args.fracao_validacao} ela admite de "
                        f"{round(args.fracao_validacao - args.tolerancia_da_celula, 4)} a "
                        f"{round(args.fracao_validacao + args.tolerancia_da_celula, 4)}. "
                        "`pendencias: []` NAO quer dizer mistura exata -- ler `por_celula`"),
                    "unidade_de_separacao": "loco (janelas que se sobrepoem, encadeadas por ligacao simples)"},
        "locos": {"total": int(len(tamanhos)), "maior": int(tamanhos.max()),
                  "com_uma_janela": int((tamanhos == 1).sum()),
                  "mediana_de_janelas": float(tamanhos.median())},
        "treino": composicao(treino),
        "validacao": composicao(validacao),
        "por_celula": por_celula,
        "disjuncao_verificada": not problemas,
        "violacoes": len(problemas),
        "pendencias": pendencias,
        "o_que_nao_prova": [
            "locos distintos ainda podem ser parecidos (parologos, repeticoes, familias de genes): isto garante "
            "ausencia de SOBREPOSICAO DE SEQUENCIA, nao independencia estatistica",
            "nao decide o tamanho da validacao, que e declarado",
        ],
        "saidas": saidas,
    }
    (out_dir / "separacao_por_locus.json").write_text(
        json.dumps(relatorio, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({k: relatorio[k] for k in
                      ("receita", "locos", "treino", "validacao", "disjuncao_verificada", "violacoes",
                       "pendencias", "saidas")}, ensure_ascii=False, indent=2))

    if pendencias:
        print("\nFALHOU: a separacao nao foi publicada (veja `pendencias`).")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
