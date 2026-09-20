#!/usr/bin/env python3
"""G4: monta o pool GLOBAL (gnomAD v4.1 joint) da metade global da mistura 60/40 do adapter.

Roda no notebook (pandas + pyarrow + pysam + credencial AWS). So le o S3; escreve em --out-dir.
Plano: docs/proposta_mosaic_regionalizacao_desenvolvimento.md, secao 5.1.

POR QUE EXISTE
--------------
O adapter populacional e treinado numa mistura de ~60% global e 40% ABraOM. O lado do ABraOM ja existe
(`audit_abraom_source.py`). Este script monta o outro lado.

E um PADRAO DE ACESSO NOVO, que o contrato do Mosaic nao cobre. O `specs/GNOMAD_S3_READ.md` descreve *lookup de
alelos ja conhecidos* -- o join pergunta "qual a AF desta variante?". Aqui a pergunta e inversa: "me de variantes".
Nada no contrato proibe isso; o que ele proibe e COPIAR o VCF (`copy_local: false`) e o que ele explica e por que
o join nao faz scan (817 GiB, e ausencia de alelo ja e `not_found`). Amostrar por regiao e escolha NOSSA, por
custo, e o vies que ela introduz entra declarado no manifesto -- nao herdado em silencio daquele contrato.

DECISOES DECLARADAS (nenhuma e default silencioso)
--------------------------------------------------
`--geografia`      obrigatorio. `uniforme` sorteia regioes proporcionais ao comprimento do cromossomo;
                   `casado_ao_abraom` sorteia proporcional a distribuicao do pool do ABraOM. O casamento REDUZ a
                   diferenca espacial grosseira entre as fontes; NAO demonstra equivalencia de contexto (dentro do
                   cromossomo continuam diferindo genes, regioes codificantes, cobertura e filtros de descoberta).
`--af-min`         opcional. Sem ele NAO ha piso. O ABraOM tem 1.171 individuos (AF minima ~4,3e-4) e o gnomAD tem
                   ~800 mil (singleton ~6e-7): tres ordens de grandeza de diferenca no piso. Cortar o lado global
                   no piso do ABraOM aproxima o INTERVALO de frequencias observadas -- nao iguala distribuicoes,
                   ancestralidades, cobertura nem processo de descoberta, e muda o que "global" significa nesta
                   campanha. Por isso o relatorio SEMPRE mede o custo do piso por bin e por cromossomo, mesmo
                   quando ele nao e aplicado: a decisao tem de ser tomada com numero, nao com argumento.
`--af-campo`       qual INFO carrega a AF. Default `AF_joint`, a agregada, por coerencia com o `gnomad_af_bin` que
                   pareia caso e controle na avaliacao. E PROPOSTA nossa: o Mosaic fixou a unidade da AVALIACAO,
                   nao a estatistica de amostragem do treino.

As regioes saem de uma semente sobre os comprimentos dos cromossomos, NUNCA dos loci do benchmark -- escolher
regiao olhando o conjunto de avaliacao seria deixar a avaliacao decidir o treino.

O QUE NAO PROVA
---------------
- Nao demonstra que as duas metades sao comparaveis; mede diferencas e declara as escolhas.
- Nao cobre a separacao populacional treino/validacao do adapter (por loci), que continua pendente.
- Nao valida sequencia: quem faz isso e `audit_variant_windows.py` sobre o plano.

USO (notebook)
--------------
    export WORK=~/testeArq/lumina-beat-regionalization
    PYTHONPATH="$WORK" python3 "$WORK"/scripts/build_global_variant_pool.py \\
        --fai ~/hg38/hg38.fa.fai --abraom-pool ~/artifacts/redesenho/g4_abraom/abraom_pool.parquet \\
        --geografia casado_ao_abraom --n-regioes 2000 --tamanho-regiao-bp 20000 \\
        --brazil-variants ~/artifacts/redesenho/g1_brazil_studies/brazil_study_variants.parquet \\
        --selection ~/artifacts/redesenho/g5_comum/selecao_comum.parquet \\
        --snapshot ~/artifacts/redesenho/g2_final_nenhum/core_head_snapshot.parquet \\
        --out-dir ~/artifacts/redesenho/g4_global
    echo "exit=$?"
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.audit_abraom_source import (  # noqa: E402
    AUTOSSOMOS, MOTIVO_OK, chave, chaves_do_parquet, classificar, distribuicao_af, e_snv)
from scripts.build_core_locus_head_snapshot import ROLE_TEST, ROLE_VALIDATION, normalize_chrom  # noqa: E402
from scripts.import_mosaic_brazil_studies import sha256_file  # noqa: E402

#: Proveniencia: `lumina-mosaic/src/mosaic/annotations/gnomad.py` (BUCKET/PREFIX/VCF_NAME/REGION).
BUCKET = "ai4bio-lumina"
PREFIX = "data/external/gnomad-joint-v4.1"
VCF_NAME = "gnomad.joint.v4.1.sites.{chrom}.vcf.bgz"
REGION = "us-east-2"
PRESIGN_TTL = 6 * 3600

COLUNA_AF = "af_gnomad"

MOTIVO_FILTRO = "filtro_do_gnomad"
MOTIVO_SEM_AF = "sem_af"
MOTIVO_AC_ZERO = "ac0"
MOTIVO_ABAIXO_DO_PISO = "abaixo_do_piso"
MOTIVO_TETO_DA_REGIAO = "teto_da_regiao"

GEOGRAFIAS = ("uniforme", "casado_ao_abraom")


# --------------------------------------------------------------------------- comprimentos e regioes

def comprimentos_do_fai(caminho: Path) -> dict[str, int]:
    """Comprimento de cada autossomo, lido do indice do FASTA. So chr1..chr22."""
    out: dict[str, int] = {}
    for linha in caminho.read_text(encoding="utf-8").splitlines():
        if not linha.strip():
            continue
        campos = linha.split("\t")
        nome = normalize_chrom(campos[0])
        if nome in AUTOSSOMOS and nome not in out:
            out[nome] = int(campos[1])
    faltando = [c for c in AUTOSSOMOS if c not in out]
    if faltando:
        raise ValueError(f"o .fai nao tem {faltando}")
    return out


def alocar_regioes(pesos: dict[str, float], *, n_regioes: int) -> dict[str, int]:
    """Reparte `n_regioes` proporcionalmente a `pesos`, por maior resto, somando exatamente n_regioes."""
    if n_regioes <= 0:
        raise ValueError(f"n_regioes tem de ser positivo, veio {n_regioes}")
    total = float(sum(pesos.values()))
    if total <= 0:
        raise ValueError("pesos somam zero")
    exatos = {c: n_regioes * peso / total for c, peso in pesos.items()}
    alocacao = {c: int(valor) for c, valor in exatos.items()}
    sobra = n_regioes - sum(alocacao.values())
    ordem = sorted(exatos, key=lambda c: (-(exatos[c] - alocacao[c]), c))
    for c in ordem[:sobra]:
        alocacao[c] += 1
    return alocacao


def sortear_regioes(
    rng: np.random.Generator, comprimentos: dict[str, int], alocacao: dict[str, int], *, tamanho_bp: int
) -> list[tuple[str, int, int]]:
    """Regioes 0-based half-open, sorteadas por semente. Nao olha nenhum loco do benchmark."""
    regioes: list[tuple[str, int, int]] = []
    for chrom in sorted(alocacao):
        quantas = alocacao[chrom]
        if quantas <= 0:
            continue
        limite = comprimentos[chrom] - tamanho_bp
        if limite <= 0:
            raise ValueError(f"{chrom} ({comprimentos[chrom]} bp) e menor que a regiao de {tamanho_bp} bp")
        inicios = np.sort(rng.integers(0, limite, size=quantas))
        regioes.extend((chrom, int(i), int(i) + tamanho_bp) for i in inicios)
    return regioes


# --------------------------------------------------------------------------- leitura do VCF

def rotulo_do_filtro(rec: Any) -> str:
    """Espelha `_filter_label` do Mosaic: sem chave nenhuma e PASS."""
    chaves = list(rec.filter.keys()) if getattr(rec, "filter", None) is not None else []
    return ";".join(chaves) if chaves else "PASS"


def _valor_a(info: Any, chave_info: str, alt_index: int) -> float | None:
    """INFO Number=A: um valor por ALT. Espelha `_info_a` do Mosaic."""
    try:
        valor = info.get(chave_info)
    except (KeyError, ValueError):
        return None
    if isinstance(valor, (list, tuple)):
        valor = valor[alt_index] if alt_index < len(valor) else None
    if valor is None:
        return None
    try:
        return float(valor)
    except (TypeError, ValueError):
        return None


def linhas_do_registro(
    rec: Any, *, af_campo: str, af_min: float | None, exigir_pass: bool = True
) -> tuple[list[dict[str, Any]], Counter]:
    """Converte um registro do VCF nas linhas de SNV utilizaveis, com o motivo de cada recusa.

    Duck-typed de proposito (`.chrom`, `.pos`, `.ref`, `.alts`, `.info`, `.filter`): assim a regra e testavel sem
    pysam e sem rede, que e a parte que erra na pratica. So `abrir_gnomad` toca a biblioteca.
    """
    motivos: Counter = Counter()
    if exigir_pass and rotulo_do_filtro(rec) != "PASS":
        motivos[MOTIVO_FILTRO] += 1
        return [], motivos

    linhas: list[dict[str, Any]] = []
    alts = rec.alts or ()
    for indice, alt in enumerate(alts):
        if not e_snv(rec.ref, alt):
            motivos["nao_snv"] += 1
            continue
        af = _valor_a(rec.info, af_campo, indice)
        if af is None:
            motivos[MOTIVO_SEM_AF] += 1
            continue
        ac = _valor_a(rec.info, "AC_joint", indice)
        if ac is not None and ac == 0:
            motivos[MOTIVO_AC_ZERO] += 1  # ADR 0003 do Mosaic: ac0 NAO e raro, e ausencia de observacao
            continue
        if af_min is not None and af < af_min:
            motivos[MOTIVO_ABAIXO_DO_PISO] += 1
            continue
        linhas.append({"chrom": normalize_chrom(rec.chrom), "pos": int(rec.pos),
                       "ref": str(rec.ref).upper(), "alt": str(alt).upper(), COLUNA_AF: af})
    return linhas, motivos


def coletar(
    fetch: Callable[[str, int, int], Iterable[Any]],
    regioes: list[tuple[str, int, int]],
    *,
    af_campo: str,
    af_min: float | None,
    max_por_regiao: int | None,
) -> tuple[pd.DataFrame, Counter]:
    """Percorre as regioes sorteadas e junta as linhas. `max_por_regiao` limita o aglomerado por desequilibrio."""
    motivos: Counter = Counter()
    coletadas: list[dict[str, Any]] = []
    for chrom, inicio, fim in regioes:
        da_regiao: list[dict[str, Any]] = []
        for rec in fetch(chrom, inicio, fim):
            linhas, parciais = linhas_do_registro(rec, af_campo=af_campo, af_min=af_min)
            motivos.update(parciais)
            da_regiao.extend(linhas)
        if max_por_regiao is not None and len(da_regiao) > max_por_regiao:
            motivos[MOTIVO_TETO_DA_REGIAO] += len(da_regiao) - max_por_regiao
            da_regiao = da_regiao[:max_por_regiao]
        coletadas.extend(da_regiao)
    if not coletadas:
        return pd.DataFrame(columns=["chrom", "pos", "ref", "alt", COLUNA_AF]), motivos
    frame = pd.DataFrame(coletadas)
    antes = len(frame)
    frame = frame.drop_duplicates(subset=["chrom", "pos", "ref", "alt"]).reset_index(drop=True)
    motivos["duplicata_entre_regioes"] += antes - len(frame)
    return frame, motivos


def abrir_gnomad(chrom: str) -> Callable[[str, int, int], Iterable[Any]]:
    """Abre o VCF do cromossomo direto no S3, sem copiar. Unica parte que depende de pysam e de rede."""
    import boto3  # import tardio: o resto do modulo roda sem AWS
    import pysam

    credenciais = boto3.Session().get_credentials()
    if credenciais is None:
        raise RuntimeError("sem credencial AWS (IMDS/env) para ler o gnomAD no S3")
    url = boto3.client("s3", region_name=REGION).generate_presigned_url(
        "get_object",
        Params={"Bucket": BUCKET, "Key": f"{PREFIX}/{VCF_NAME.format(chrom=chrom)}"},
        ExpiresIn=PRESIGN_TTL)
    handle = pysam.VariantFile(url)
    return lambda c, inicio, fim: handle.fetch(c, inicio, fim)


def fetch_por_cromossomo(regioes: list[tuple[str, int, int]]) -> Callable[[str, int, int], Iterable[Any]]:
    """Um handle por cromossomo, reaproveitado entre as regioes dele."""
    handles: dict[str, Callable[[str, int, int], Iterable[Any]]] = {}

    def fetch(chrom: str, inicio: int, fim: int) -> Iterable[Any]:
        if chrom not in handles:
            handles[chrom] = abrir_gnomad(chrom)
        return handles[chrom](chrom, inicio, fim)

    return fetch


# --------------------------------------------------------------------------- medidas

def custo_do_piso(frame: pd.DataFrame, af_min: float) -> dict[str, Any]:
    """Quanto um piso de AF removeria, por bin e por cromossomo -- medido mesmo quando o piso NAO e aplicado."""
    if frame.empty:
        return {"piso": af_min, "removeria": 0}
    abaixo = frame[frame[COLUNA_AF] < af_min]
    return {
        "piso": af_min,
        "removeria": int(len(abaixo)),
        "fracao": round(len(abaixo) / len(frame), 4),
        "por_bin": distribuicao_af(abaixo[COLUNA_AF]) if len(abaixo) else {},
        "por_cromossomo": {str(c): int(n) for c, n in abaixo["chrom"].value_counts().sort_index().items()},
        "o_que_nao_faz": ("aproxima o intervalo de frequencias observadas; nao iguala distribuicoes, "
                          "ancestralidades, cobertura nem processo de descoberta"),
    }


def comparar_geografia(pool: pd.DataFrame, abraom: pd.DataFrame) -> dict[str, Any]:
    """Fracao por cromossomo nas duas metades, e a maior diferenca -- o tamanho do confundimento espacial."""
    a = abraom["chrom"].value_counts(normalize=True)
    g = pool["chrom"].value_counts(normalize=True) if len(pool) else a * 0
    cromossomos = sorted(set(a.index) | set(g.index))
    diferencas = {c: round(float(g.get(c, 0.0) - a.get(c, 0.0)), 4) for c in cromossomos}
    pior = max(diferencas, key=lambda c: abs(diferencas[c])) if diferencas else None
    return {"diferenca_por_cromossomo": diferencas,
            "maior_diferenca": {"cromossomo": pior, "valor": diferencas.get(pior) if pior else None}}


# --------------------------------------------------------------------------- main

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fai", required=True, type=Path, help="indice do hg38, para os comprimentos")
    parser.add_argument("--abraom-pool", required=True, type=Path, help="a outra metade: piso e geografia")
    parser.add_argument("--geografia", required=True, choices=GEOGRAFIAS)
    parser.add_argument("--af-campo", default="AF_joint")
    parser.add_argument("--af-min", type=float, help="sem ele NAO ha piso; o custo e medido de todo jeito")
    parser.add_argument("--n-regioes", type=int, default=2000)
    parser.add_argument("--tamanho-regiao-bp", type=int, default=20000)
    parser.add_argument("--max-por-regiao", type=int, default=50)
    parser.add_argument("--brazil-variants", type=Path)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--snapshot", action="append", type=Path, default=None)
    parser.add_argument("--no-reserve-chr8", action="store_true")
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args(argv)

    exclusoes_ausentes: list[str] = []
    if not args.brazil_variants:
        exclusoes_ausentes.append("--brazil-variants: membros dos dois estudos brasileiros (G1)")
    if not args.selection:
        exclusoes_ausentes.append("--selection: conjunto de selecao comum (G5)")
    if not (args.snapshot or []):
        exclusoes_ausentes.append("--snapshot: validacao e teste da cabeca (G2)")

    comprimentos = comprimentos_do_fai(args.fai.expanduser())
    abraom = pd.read_parquet(args.abraom_pool.expanduser())
    piso_do_abraom = float(abraom["af_abraom"].min())

    reservar_chr8 = not args.no_reserve_chr8
    elegiveis = {c: v for c, v in comprimentos.items() if not (reservar_chr8 and c == "chr8")}
    if args.geografia == "casado_ao_abraom":
        contagem = abraom["chrom"].value_counts()
        pesos = {c: float(contagem.get(c, 0)) for c in elegiveis}
        pesos = {c: v for c, v in pesos.items() if v > 0}
    else:
        pesos = {c: float(v) for c, v in elegiveis.items()}

    alocacao = alocar_regioes(pesos, n_regioes=args.n_regioes)
    rng = np.random.default_rng(args.seed)
    regioes = sortear_regioes(rng, comprimentos, alocacao, tamanho_bp=args.tamanho_regiao_bp)

    bruto, motivos_leitura = coletar(fetch_por_cromossomo(regioes), regioes, af_campo=args.af_campo,
                                     af_min=args.af_min, max_por_regiao=args.max_por_regiao)

    identidades: dict[str, str] = {str(args.abraom_pool): sha256_file(args.abraom_pool.expanduser())}
    membros: set[str] = set()
    if args.brazil_variants:
        membros = chaves_do_parquet(args.brazil_variants.expanduser())
        identidades[str(args.brazil_variants)] = sha256_file(args.brazil_variants.expanduser())
    avaliacao: set[str] = set()
    if args.selection:
        avaliacao |= chaves_do_parquet(args.selection.expanduser())
        identidades[str(args.selection)] = sha256_file(args.selection.expanduser())
    for caminho in (args.snapshot or []):
        avaliacao |= chaves_do_parquet(caminho.expanduser(), papeis=(ROLE_VALIDATION, ROLE_TEST))
        identidades[str(caminho)] = sha256_file(caminho.expanduser())

    if len(bruto):
        motivos = classificar(bruto, membros=membros, alelos_de_avaliacao=avaliacao,
                              reservar_chr8=reservar_chr8, coluna_af=COLUNA_AF)
        pool = bruto[motivos == MOTIVO_OK].reset_index(drop=True)
        por_motivo = {m: int(q) for m, q in motivos.value_counts().items()}
    else:
        pool, por_motivo = bruto, {}

    pode_publicar = not exclusoes_ausentes and len(pool) > 0
    out_dir = args.out_dir.expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    saidas: dict[str, Any] = {}
    if pode_publicar:
        pool_path = out_dir / "global_pool.parquet"
        pool.to_parquet(pool_path, index=False)
        saidas = {"pool": str(pool_path), "pool_sha256": sha256_file(pool_path)}

    pendencias = list(exclusoes_ausentes)
    if len(pool) == 0:
        pendencias.append("nenhuma variante sobrou: aumentar --n-regioes ou revisar os filtros")

    relatorio: dict[str, Any] = {
        "receita": {
            "af_campo": args.af_campo, "af_min_aplicado": args.af_min,
            "geografia": args.geografia, "n_regioes": args.n_regioes,
            "tamanho_regiao_bp": args.tamanho_regiao_bp, "max_por_regiao": args.max_por_regiao,
            "seed": args.seed, "chr8_reservado": reservar_chr8,
            "regioes_sorteadas": len(regioes),
            "bases_lidas": len(regioes) * args.tamanho_regiao_bp,
            "origem_das_regioes": "semente sobre os comprimentos do .fai; nenhum loco do benchmark participa",
            "padrao_de_acesso": ("amostragem por regiao via .tbi, leitura remota sem copia; e um acesso NOVO, "
                                 "nao coberto pelo contrato de lookup do Mosaic"),
        },
        "entradas": {"fai": str(args.fai), "abraom_pool": str(args.abraom_pool)},
        "identidades_das_entradas": identidades,
        "motivos_de_leitura": dict(sorted(motivos_leitura.items())),
        "por_motivo_de_exclusao": por_motivo,
        "pool": {
            "n": int(len(pool)),
            "por_cromossomo": {str(c): int(n) for c, n in pool["chrom"].value_counts().sort_index().items()}
            if len(pool) else {},
            "por_bin": distribuicao_af(pool[COLUNA_AF]) if len(pool) else {},
        },
        "piso_do_abraom": piso_do_abraom,
        "custo_de_casar_o_piso": custo_do_piso(pool, piso_do_abraom),
        "geografia_contra_o_abraom": comparar_geografia(pool, abraom),
        "pronto_para_amostrar": pode_publicar,
        "pendencias": pendencias,
        "falta_antes_de_treinar": [
            "decidir se o piso de AF entra (o custo esta medido acima)",
            "separacao populacional treino/validacao do adapter, por loci",
            "peso da loss entre posicoes de variante e de referencia",
        ],
        "o_que_nao_prova": [
            "casar cromossomo reduz diferenca espacial grosseira; nao demonstra equivalencia de contexto",
            "casar o piso aproxima o intervalo de AF; nao iguala distribuicoes nem ancestralidades",
            "a amostragem por regiao tem vies proprio: variantes do mesmo bloco chegam juntas",
        ],
        "saidas": saidas,
    }
    (out_dir / "pool_global.json").write_text(
        json.dumps(relatorio, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({k: relatorio[k] for k in
                      ("receita", "motivos_de_leitura", "por_motivo_de_exclusao", "pool",
                       "custo_de_casar_o_piso", "geografia_contra_o_abraom", "pronto_para_amostrar",
                       "pendencias", "saidas")}, ensure_ascii=False, indent=2))
    return 0 if pode_publicar else 2


if __name__ == "__main__":
    sys.exit(main())
