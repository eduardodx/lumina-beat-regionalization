#!/usr/bin/env python3
"""Plano de janelas do adapter populacional: quais variantes, onde na janela, quais spans mascarados.

Roda no notebook (pandas + numpy + pyarrow; sem GPU). So le os pools; escreve em --out-dir.
Plano: docs/proposta_mosaic_regionalizacao_desenvolvimento.md, secao 5.1 (gate G4).

POR QUE UM PLANO, E NAO AS SEQUENCIAS
-------------------------------------
Materializar milhoes de janelas de 4.096 bp custaria dezenas de GB e congelaria o FASTA dentro do artefato. O
plano guarda a DECISAO -- variante, deslocamento da janela, spans -- em poucas colunas, e o treinador monta a
sequencia na hora com `eval/embedding_probe/windows.py`, que ja confere REF contra o FASTA e descarta janela com
base fora de ACGT. Assim o artefato e pequeno, reprodutivel pela seed e auditavel linha a linha.

A RECEITA (declarada na secao 5.1, antes de qualquer treino)
------------------------------------------------------------
- Uma variante focal por janela: torna equivalentes as tres leituras da mistura e permite auditar a proporcao.
- Posicao da variante: a JANELA e deslocada em torno dela, preservando a coordenada genomica e o contexto real.
  A mutacao nunca e transportada para outro lugar da sequencia.
- Margem minima das bordas: a leitura da cabeca usa contexto local de +-64 bp, entao o focal precisa ter esse
  contexto dentro da janela.
- Mistura: fracao de JANELAS por fonte (0,6 global / 0,4 ABraOM). 1,0 global e o braco MG da ablacao.
- Amostragem estratificada por bin de AF: no ABraOM, 54% das variantes estao no bin mais raro, entao amostrar
  uniformemente faria o adapter ver quase so singletons.
- Spans: um cobrindo a variante, mais uma fracao de spans so em posicoes de referencia -- sem eles, mascarar
  passa a coincidir com "aqui ha variante", risco que o piloto deve investigar.

O QUE NAO PROVA
---------------
- Nao treina nada e nao le o FASTA: janela invalida so aparece no gerador de sequencia, e o plano registra isso
  como pendencia do piloto.
- Sem o pool global, o plano sai 100% ABraOM e marcado como NAO pronto para a campanha -- serve de smoke.

USO (notebook)
--------------
    export WORK=~/testeArq/lumina-beat-regionalization
    PYTHONPATH="$WORK" python3 "$WORK"/scripts/build_adapter_window_plan.py \\
        --abraom-pool ~/artifacts/redesenho/g4_abraom/abraom_pool.parquet \\
        --n-janelas 20000 --seed 20260920 \\
        --out-dir ~/artifacts/redesenho/g4_plano_smoke
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

from scripts.audit_abraom_source import AF_BINS  # noqa: E402
from scripts.import_mosaic_brazil_studies import sha256_file  # noqa: E402

FONTE_ABRAOM = "abraom"
FONTE_GLOBAL = "global"

#: A leitura da cabeca usa contexto local de +-64 bp em torno do focal (RegimeAHead).
MARGEM_PADRAO = 64

TIPO_VARIANTE = "variante"
TIPO_REFERENCIA = "referencia"


def rotular_bins(af: pd.Series) -> pd.Series:
    return pd.cut(af, bins=list(AF_BINS), include_lowest=True, right=True).astype(str)


def amostrar_estratificado(pool: pd.DataFrame, *, n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Reparte `n` igualmente entre os bins de AF; bin curto entrega o que tem e o resto se redistribui.

    Deterministico dada a seed: o pool e ordenado por coordenada antes de sortear.
    """
    if n <= 0:
        return pool.iloc[0:0].copy()
    ordenado = pool.sort_values(["chrom", "pos", "ref", "alt"], kind="mergesort").reset_index(drop=True)
    ordenado["af_bin"] = rotular_bins(ordenado["af_abraom"])

    # Do bin mais CURTO para o mais farto: assim o que um bin curto nao consegue entregar sobra para quem tem
    # folga, em vez de se perder. Percorrer na ordem contraria deixaria o total abaixo do pedido.
    grupos = [(str(nome), sub) for nome, sub in ordenado.groupby("af_bin", sort=True)]
    grupos.sort(key=lambda item: (len(item[1]), item[0]))

    escolhidos: list[pd.DataFrame] = []
    faltando = n
    restantes = len(grupos)
    for _, disponivel in grupos:
        alvo_por_bin = faltando // restantes if restantes else 0
        quantidade = min(alvo_por_bin, len(disponivel))
        if quantidade:
            indices = rng.choice(len(disponivel), size=quantidade, replace=False)
            escolhidos.append(disponivel.iloc[np.sort(indices)])
        faltando -= quantidade
        restantes -= 1
    if not escolhidos:
        return ordenado.iloc[0:0].copy()
    return pd.concat(escolhidos, ignore_index=True)


def posicoes_focais(rng: np.random.Generator, *, quantidade: int, window_bp: int, margem: int) -> np.ndarray:
    """Indice do focal dentro da janela, uniforme em [margem, window_bp - margem)."""
    if window_bp <= 2 * margem:
        raise ValueError(f"janela de {window_bp} bp nao comporta margem de {margem} bp dos dois lados")
    return rng.integers(margem, window_bp - margem, size=quantidade)


def spans_da_janela(
    rng: np.random.Generator, *, focal_index: int, window_bp: int,
    span_min: int, span_max: int, spans_de_referencia: int, margem: int,
) -> list[tuple[int, int, str]]:
    """Um span cobrindo o focal, mais `spans_de_referencia` que NAO tocam o focal."""
    comprimento = int(rng.integers(span_min, span_max + 1))
    deslocamento = int(rng.integers(0, comprimento))
    inicio = max(0, min(focal_index - deslocamento, window_bp - comprimento))
    spans = [(inicio, inicio + comprimento, TIPO_VARIANTE)]

    tentativas = 0
    while len([s for s in spans if s[2] == TIPO_REFERENCIA]) < spans_de_referencia and tentativas < 100:
        tentativas += 1
        tamanho = int(rng.integers(span_min, span_max + 1))
        comeco = int(rng.integers(0, max(1, window_bp - tamanho)))
        fim = comeco + tamanho
        if comeco <= focal_index < fim:
            continue  # tocaria o focal: seria outro span de variante, nao de referencia
        if any(comeco < outro_fim and outro_inicio < fim for outro_inicio, outro_fim, _ in spans):
            continue  # sobreposto a outro span
        spans.append((comeco, fim, TIPO_REFERENCIA))
    return spans


def montar_plano(
    amostras: pd.DataFrame, *, fonte: str, rng: np.random.Generator, window_bp: int, margem: int,
    span_min: int, span_max: int, spans_de_referencia: int,
) -> pd.DataFrame:
    if amostras.empty:
        return pd.DataFrame(columns=["variant_id", "chrom", "pos_1based", "ref", "alt", "af", "fonte",
                                     "af_bin", "focal_index", "window_start", "spans"])
    focais = posicoes_focais(rng, quantidade=len(amostras), window_bp=window_bp, margem=margem)
    linhas = []
    for (_, variante), focal in zip(amostras.iterrows(), focais):
        spans = spans_da_janela(rng, focal_index=int(focal), window_bp=window_bp, span_min=span_min,
                                span_max=span_max, spans_de_referencia=spans_de_referencia, margem=margem)
        chrom, pos = variante["chrom"], int(variante["pos"])
        ref, alt = str(variante["ref"]).upper(), str(variante["alt"]).upper()
        linhas.append({
            # Vocabulario da campanha: `variant_id` e `pos_1based`, como no snapshot e nos estudos. O ABraOM nao
            # tem variant_id do Mosaic, entao a chave e a coordenada normalizada -- estavel e comparavel.
            "variant_id": f"{chrom}:{pos}:{ref}:{alt}",
            "chrom": chrom, "pos_1based": pos, "ref": ref, "alt": alt,
            "af": float(variante["af_abraom"]), "fonte": fonte,
            "af_bin": variante.get("af_bin"),
            "focal_index": int(focal),
            # A janela e deslocada em torno da variante: a coordenada genomica dela nao muda.
            "window_start": pos - 1 - int(focal),
            "spans": json.dumps(spans),
        })
    return pd.DataFrame(linhas)


def resumo_do_plano(plano: pd.DataFrame, *, window_bp: int) -> dict[str, Any]:
    if plano.empty:
        return {"n": 0}
    spans = [json.loads(s) for s in plano["spans"]]
    comprimentos = [fim - inicio for janela in spans for inicio, fim, _ in janela]
    de_variante = [s for janela in spans for s in janela if s[2] == TIPO_VARIANTE]
    de_referencia = [s for janela in spans for s in janela if s[2] == TIPO_REFERENCIA]
    return {
        "n": int(len(plano)),
        "por_fonte": {str(f): int(q) for f, q in plano["fonte"].value_counts().items()},
        "fracao_global": round(float((plano["fonte"] == FONTE_GLOBAL).mean()), 4),
        "por_af_bin": {str(b): int(q) for b, q in plano["af_bin"].value_counts().sort_index().items()},
        "por_cromossomo": {str(c): int(q) for c, q in plano["chrom"].value_counts().sort_index().items()},
        "focal_index": {
            "min": int(plano["focal_index"].min()), "max": int(plano["focal_index"].max()),
            "mediana": float(plano["focal_index"].median()),
            "centro_da_janela": window_bp // 2 - 1,
        },
        "spans": {
            "por_janela": round(float(len(comprimentos) / len(plano)), 3),
            "de_variante": len(de_variante), "de_referencia": len(de_referencia),
            "comprimento_min": min(comprimentos), "comprimento_max": max(comprimentos),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--abraom-pool", required=True, type=Path)
    parser.add_argument("--global-pool", type=Path, help="pool do gnomAD; sem ele o plano e so smoke")
    parser.add_argument("--n-janelas", type=int, required=True)
    parser.add_argument("--fracao-global", type=float, default=0.6)
    parser.add_argument("--window-bp", type=int, default=4096)
    parser.add_argument("--margem", type=int, default=MARGEM_PADRAO)
    parser.add_argument("--span-min", type=int, default=3)
    parser.add_argument("--span-max", type=int, default=10)
    parser.add_argument("--spans-de-referencia", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args(argv)

    if not 0.0 <= args.fracao_global <= 1.0:
        print(f"FALHOU: --fracao-global {args.fracao_global} fora de [0,1]")
        return 2
    if args.span_min < 1 or args.span_max < args.span_min:
        print(f"FALHOU: spans invalidos ({args.span_min}, {args.span_max})")
        return 2

    abraom = pd.read_parquet(args.abraom_pool.expanduser())
    tem_global = args.global_pool is not None
    fracao_global = args.fracao_global if tem_global else 0.0
    n_global = int(round(args.n_janelas * fracao_global))
    n_abraom = args.n_janelas - n_global

    rng = np.random.default_rng(args.seed)
    amostras_abraom = amostrar_estratificado(abraom, n=n_abraom, rng=rng)
    plano = montar_plano(amostras_abraom, fonte=FONTE_ABRAOM, rng=rng, window_bp=args.window_bp,
                         margem=args.margem, span_min=args.span_min, span_max=args.span_max,
                         spans_de_referencia=args.spans_de_referencia)
    if tem_global:
        glob = pd.read_parquet(args.global_pool.expanduser())
        amostras_global = amostrar_estratificado(glob, n=n_global, rng=rng)
        plano = pd.concat([plano, montar_plano(
            amostras_global, fonte=FONTE_GLOBAL, rng=rng, window_bp=args.window_bp, margem=args.margem,
            span_min=args.span_min, span_max=args.span_max,
            spans_de_referencia=args.spans_de_referencia)], ignore_index=True)

    out_dir = args.out_dir.expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    plano_path = out_dir / "plano_de_janelas.parquet"
    plano.to_parquet(plano_path, index=False)

    manifesto: dict[str, Any] = {
        "receita": {
            "window_bp": args.window_bp, "margem": args.margem,
            "variantes_por_janela": 1,
            "fracao_global_pedida": args.fracao_global, "fracao_global_efetiva": fracao_global,
            "span_min": args.span_min, "span_max": args.span_max,
            "spans_de_referencia": args.spans_de_referencia,
            "seed": args.seed,
            "posicao_da_variante": "janela deslocada em torno da variante; a coordenada genomica nao muda",
            "amostragem": "estratificada por bin de AF, repartindo igualmente e redistribuindo bin curto",
        },
        "entradas": {
            "abraom_pool": str(args.abraom_pool), "abraom_pool_sha256": sha256_file(args.abraom_pool.expanduser()),
            "global_pool": str(args.global_pool) if tem_global else None,
            "global_pool_sha256": sha256_file(args.global_pool.expanduser()) if tem_global else None,
        },
        "resumo": resumo_do_plano(plano, window_bp=args.window_bp),
        "pronto_para_campanha": tem_global,
        "pendencias": [] if tem_global else [
            "sem pool global: o plano e 100% ABraOM e serve de smoke, nao da campanha"
        ],
        "o_que_nao_prova": [
            "o plano nao le o FASTA: janela invalida so aparece no gerador de sequencia",
            "spans de referencia sao guarda contra o atalho 'mascarado = variante', ainda a investigar no piloto",
        ],
        "saidas": {"plano": str(plano_path), "plano_sha256": sha256_file(plano_path)},
    }
    (out_dir / "manifesto_do_plano.json").write_text(
        json.dumps(manifesto, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({k: manifesto[k] for k in ("resumo", "pronto_para_campanha", "pendencias", "saidas")},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
