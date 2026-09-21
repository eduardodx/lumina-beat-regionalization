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

from eval.embedding_probe.windows import WindowError, build_window  # noqa: E402
from scripts.audit_variant_windows import abrir_fasta  # noqa: E402

from scripts.audit_abraom_source import AF_BINS  # noqa: E402
from scripts.import_mosaic_brazil_studies import sha256_file  # noqa: E402

FONTE_ABRAOM = "abraom"
FONTE_GLOBAL = "global"

#: A leitura da cabeca usa contexto local de +-64 bp em torno do focal (RegimeAHead).
MARGEM_PADRAO = 64

TIPO_VARIANTE = "variante"

# Quanto a mistura medida nas linhas pode se afastar da pedida antes de reprovar. Nao e folga de desenho: e
# arredondamento de 60/40 em N janelas.
TOLERANCIA_DA_MISTURA = 0.01
TIPO_REFERENCIA = "referencia"


#: Como cada metade da mistura chama a sua frequencia. Aceitar as duas e o que permite ao mesmo gerador montar
#: o plano do ABraOM e o do gnomAD sem renomear coluna -- renomear seria a forma mais facil de trocar as fontes.
COLUNAS_DE_AF = ("af_abraom", "af_gnomad")


def coluna_de_af(pool: pd.DataFrame) -> str:
    """Qual coluna carrega a AF deste pool. Falha alto se nenhuma ou as duas aparecerem."""
    presentes = [c for c in COLUNAS_DE_AF if c in pool.columns]
    if len(presentes) != 1:
        raise ValueError(f"esperava exatamente uma de {COLUNAS_DE_AF}, achei {presentes or list(pool.columns)}")
    return presentes[0]


def rotular_bins(af: pd.Series) -> pd.Series:
    return pd.cut(af, bins=list(AF_BINS), include_lowest=True, right=True).astype(str)


def amostrar_estratificado(pool: pd.DataFrame, *, n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Reparte `n` igualmente entre os bins de AF; bin curto entrega o que tem e o resto se redistribui.

    Deterministico dada a seed: o pool e ordenado por coordenada antes de sortear.
    """
    if n <= 0:
        return pool.iloc[0:0].copy()
    ordenado = pool.sort_values(["chrom", "pos", "ref", "alt"], kind="mergesort").reset_index(drop=True)
    ordenado["af_bin"] = rotular_bins(ordenado[coluna_de_af(ordenado)])

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
    coluna = coluna_de_af(amostras)
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
            "af": float(variante[coluna]), "fonte": fonte,
            "af_bin": variante.get("af_bin"),
            "focal_index": int(focal),
            # A janela e deslocada em torno da variante: a coordenada genomica dela nao muda.
            "window_start": pos - 1 - int(focal),
            "spans": json.dumps(spans),
        })
    return pd.DataFrame(linhas)


def indices_invalidos(plano: pd.DataFrame, fetch, *, window_bp: int) -> list[int]:
    """Posicoes do plano cuja janela DECLARADA nao se constroi. Mesma regra do auditor de janelas."""
    ruins: list[int] = []
    for posicao, linha in enumerate(plano.itertuples(index=False)):
        try:
            build_window(fetch, chrom=str(linha.chrom), pos_1based=int(linha.pos_1based),
                         ref=str(linha.ref), alt=str(linha.alt), window_bp=window_bp,
                         focal_index=int(linha.focal_index))
        except WindowError:
            ruins.append(posicao)
    return ruins


def reparar_plano(
    plano: pd.DataFrame, pools: dict[str, pd.DataFrame], fetch, *, rng: np.random.Generator,
    window_bp: int, margem: int, span_min: int, span_max: int, spans_de_referencia: int,
    max_rodadas: int = 5,
) -> tuple[pd.DataFrame, dict[str, int], pd.DataFrame]:
    """Repoe cada janela invalida por outra DA MESMA FONTE E DO MESMO BIN DE AF.

    Por que repor em vez de descartar: descartar encolhe o plano e desloca a mistura e a estratificacao, que sao
    justamente o que a receita declara. Repor dentro do estrato mantem os dois exatos.

    O VIES QUE ISSO INTRODUZ, declarado: variantes cuja janela de 4.096 bp contem base fora de ACGT ficam
    sistematicamente de fora. E pequeno (42 em 50.000 na rodada de 21/09) e inevitavel -- o modelo nao consegue
    ler essa janela de qualquer forma --, mas e vies, nao neutralidade.

    Devolve (plano reparado, substituicoes por fonte, linhas que nao deu para repor).
    """
    substituicoes: dict[str, int] = {}
    for _ in range(max_rodadas):
        ruins = indices_invalidos(plano, fetch, window_bp=window_bp)
        if not ruins:
            return plano, substituicoes, plano.iloc[0:0].copy()
        maus = plano.iloc[ruins].copy()
        plano = plano.drop(plano.index[ruins]).reset_index(drop=True)
        usados = set(plano["variant_id"]) | set(maus["variant_id"])
        repostos: list[pd.DataFrame] = []
        for (fonte, af_bin), grupo in maus.groupby(["fonte", "af_bin"], sort=True):
            pool = pools[str(fonte)]
            candidatos = pool[(pool["af_bin"] == af_bin) & (~pool["variant_id"].isin(usados))]
            if candidatos.empty:
                continue
            quantidade = min(len(grupo), len(candidatos))
            escolhidos = rng.choice(len(candidatos), size=quantidade, replace=False)
            amostra = candidatos.iloc[np.sort(escolhidos)]
            usados |= set(amostra["variant_id"])
            repostos.append(montar_plano(amostra, fonte=str(fonte), rng=rng, window_bp=window_bp,
                                         margem=margem, span_min=span_min, span_max=span_max,
                                         spans_de_referencia=spans_de_referencia))
            substituicoes[str(fonte)] = substituicoes.get(str(fonte), 0) + quantidade
        if not repostos:
            return plano, substituicoes, maus
        plano = pd.concat([plano, *repostos], ignore_index=True)

    restantes = plano.iloc[indices_invalidos(plano, fetch, window_bp=window_bp)].copy()
    if len(restantes):
        plano = plano.drop(restantes.index).reset_index(drop=True)
    return plano, substituicoes, restantes


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
    parser.add_argument("--fasta", type=Path,
                        help="com ele, o plano confere cada janela e repoe a invalida dentro do mesmo estrato")
    parser.add_argument("--max-rodadas-de-reparo", type=int, default=5)
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

    # O amostrador entrega o que o pool tem, sem inventar: se um bin faltou, o TOTAL e a MISTURA saem diferentes
    # do pedido -- em silencio. Medimos as duas coisas nas linhas produzidas, nao nos numeros pedidos.
    # (recalculado abaixo, depois do reparo contra o FASTA)
    produzido = int(len(plano))
    contagem = plano["fonte"].value_counts().to_dict() if produzido else {}
    n_global_produzido = int(contagem.get(FONTE_GLOBAL, 0))
    fracao_efetiva = round(n_global_produzido / produzido, 4) if produzido else 0.0

    bloqueios: list[str] = []
    if not tem_global:
        bloqueios.append("sem pool global: o plano e 100% ABraOM e serve de smoke, nao da campanha")
    if produzido != args.n_janelas:
        bloqueios.append(f"plano com {produzido} janelas, {args.n_janelas} pedidas: algum bin de AF nao tinha "
                         f"variantes suficientes")
    if tem_global and abs(fracao_efetiva - args.fracao_global) > TOLERANCIA_DA_MISTURA:
        bloqueios.append(f"mistura efetiva {fracao_efetiva} fora de {args.fracao_global} "
                         f"+/- {TOLERANCIA_DA_MISTURA}: os pools tem capacidades diferentes")

    substituicoes: dict[str, int] = {}
    irrecuperaveis = plano.iloc[0:0].copy()
    leitor = None
    if args.fasta:
        fetch, leitor = abrir_fasta(args.fasta.expanduser())
        pools: dict[str, pd.DataFrame] = {}
        for fonte, bruto in ((FONTE_ABRAOM, abraom), (FONTE_GLOBAL, glob if tem_global else None)):
            if bruto is None:
                continue
            anotado = bruto.copy()
            anotado["af_bin"] = rotular_bins(anotado[coluna_de_af(anotado)])
            anotado["variant_id"] = [f"{c}:{int(pos)}:{str(r).upper()}:{str(a).upper()}" for c, pos, r, a in
                                     zip(anotado["chrom"], anotado["pos"], anotado["ref"], anotado["alt"])]
            pools[fonte] = anotado
        plano, substituicoes, irrecuperaveis = reparar_plano(
            plano, pools, fetch, rng=rng, window_bp=args.window_bp, margem=args.margem,
            span_min=args.span_min, span_max=args.span_max,
            spans_de_referencia=args.spans_de_referencia, max_rodadas=args.max_rodadas_de_reparo)

    out_dir = args.out_dir.expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    plano_path = out_dir / "plano_de_janelas.parquet"
    plano.to_parquet(plano_path, index=False)

    # Recontagem depois do reparo: e o plano PUBLICADO que tem de bater com o pedido.
    produzido = int(len(plano))
    contagem = plano["fonte"].value_counts().to_dict() if produzido else {}
    n_global_produzido = int(contagem.get(FONTE_GLOBAL, 0))
    fracao_efetiva = round(n_global_produzido / produzido, 4) if produzido else 0.0
    bloqueios = [b for b in bloqueios if "plano com" not in b and "mistura efetiva" not in b]
    if produzido != args.n_janelas:
        bloqueios.append(f"plano com {produzido} janelas, {args.n_janelas} pedidas: algum bin de AF nao tinha "
                         f"variantes suficientes")
    if tem_global and abs(fracao_efetiva - args.fracao_global) > TOLERANCIA_DA_MISTURA:
        bloqueios.append(f"mistura efetiva {fracao_efetiva} fora de {args.fracao_global} "
                         f"+/- {TOLERANCIA_DA_MISTURA}: os pools tem capacidades diferentes")
    if len(irrecuperaveis):
        bloqueios.append(f"{len(irrecuperaveis)} janelas invalidas sem reposicao no estrato")

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
        "conferencia": {
            "janelas_pedidas": args.n_janelas, "janelas_produzidas": produzido,
            "fracao_global_efetiva_nas_linhas": fracao_efetiva,
            "tolerancia_da_mistura": TOLERANCIA_DA_MISTURA,
            "conferido_contra_o_fasta": str(args.fasta) if args.fasta else None,
            "leitor_do_fasta": leitor,
            "substituicoes_por_fonte": substituicoes,
            "janelas_sem_reposicao": int(len(irrecuperaveis)),
            "vies_da_reposicao": (
                "variantes cuja janela contem base fora de ACGT ficam sistematicamente de fora; inevitavel, "
                "porque o modelo nao le essa janela, mas e vies e nao neutralidade") if args.fasta else None,
        },
        "pronto_para_campanha": not bloqueios,
        "pendencias": bloqueios,
        "falta_antes_de_treinar": [
            "auditar ESTE plano contra o FASTA (audit_variant_windows.py le focal_index, window_start e spans)"
            if not args.fasta else "reauditar o plano reparado, para confirmar que a reposicao fechou",
            "declarar a separacao populacional entre treino e validacao do adapter, por loci",
            "fixar o peso da loss entre posicoes de variante e de referencia",
        ],
        "o_que_nao_prova": [
            "o plano nao le o FASTA: janela invalida so aparece na auditoria de janelas",
            "spans de referencia sao guarda contra o atalho 'mascarado = variante', ainda a investigar no piloto",
            "a conferencia olha total e mistura, nao a representatividade das fontes",
        ],
        "saidas": {"plano": str(plano_path), "plano_sha256": sha256_file(plano_path)},
    }
    (out_dir / "manifesto_do_plano.json").write_text(
        json.dumps(manifesto, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({k: manifesto[k] for k in ("resumo", "conferencia", "pronto_para_campanha", "pendencias",
                                                "saidas")}, ensure_ascii=False, indent=2))
    # Sem pool global o plano e smoke declarado e isso nao e falha. Total ou mistura errados, sim: o plano pedido
    # nao foi o produzido, e seguir com ele mudaria o desenho sem ninguem decidir.
    if produzido != args.n_janelas or (tem_global and abs(fracao_efetiva - args.fracao_global) > TOLERANCIA_DA_MISTURA):
        print("\nFALHOU: o plano produzido nao e o plano pedido (veja `conferencia`).")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
