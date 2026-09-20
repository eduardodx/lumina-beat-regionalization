#!/usr/bin/env python3
"""Audita o ABraOM do source-lock e publica o pool de amostragem do adapter, com identidade.

Roda no notebook (pandas + pyarrow; sem GPU). So le; escreve em --out-dir.
Plano: docs/proposta_mosaic_regionalizacao_desenvolvimento.md, secoes 4.3 e 5.1 (gate G4).

POR QUE EXISTE
--------------
O arquivo chegou conferido por sha256, mas conferir identidade nao e conhecer o conteudo. Antes de gerar janelas
sinteticas com ele, tres coisas precisam de numero:

1. **Convencao de cromossomo.** O TSV usa `1`; o snapshot e o FASTA usam `chr1`. Juntar sem normalizar daria
   zero sobreposicao em silencio -- o pior tipo de erro, porque parece "nenhum vazamento".
2. **O que nao vira janela.** A extracao e so-SNV e a suite do Mosaic e so autossomo. Linha nao-SNV ou fora de
   chr1..chr22 tem de ser contada e descartada explicitamente, nao encontrada no meio do treino.
3. **Sobreposicao com o estudo.** O `br_population_observed` e, por definicao, gold presente no ABraOM: esses
   alelos estao neste arquivo. A regra da secao 4.3 e que NENHUM alelo dos estudos entra nas janelas do adapter.
   Aqui se mede quantos sao e se produz o pool ja sem eles.

O QUE NAO PROVA
---------------
- Nao valida o ABraOM cientificamente: 1,37 milhao de variantes e pouco para 1.171 genomas completos, entao o
  "clean" e um subconjunto curado. O que isso significa para a amostragem e questao de desenho, declarada no
  plano, nao resolvida aqui.
- Nao decide chr8: mede o custo de reserva-lo, a decisao e do Eduardo (decisao E).

USO (notebook)
--------------
    export WORK=~/testeArq/lumina-beat-regionalization
    PYTHONPATH="$WORK" python3 "$WORK"/scripts/audit_abraom_source.py \\
        --abraom ~/artifacts/redesenho/g0_fontes/SABE1171.Abraom.clean.tsv \\
        --brazil-variants ~/artifacts/redesenho/g1_brazil_studies/brazil_study_variants.parquet \\
        --snapshot ~/artifacts/redesenho/g2_final_nenhum/core_head_snapshot.parquet \\
        --out-dir ~/artifacts/redesenho/g4_abraom
    echo "exit=$?"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_core_locus_head_snapshot import normalize_chrom  # noqa: E402
from scripts.import_mosaic_brazil_studies import sha256_file  # noqa: E402

COLUNAS = ("chrom", "pos", "ref", "alt", "af_abraom")
BASES = frozenset("ACGT")

#: Autossomos: a suite do Mosaic e so chr1..chr22 (`mosaic.reference.PRIMARY_CHROMS`).
AUTOSSOMOS = tuple(f"chr{i}" for i in range(1, 23))

#: Bins de AF declarados, usados na amostragem estratificada da secao 5.1.
AF_BINS = (0.0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0)

MOTIVO_OK = "ok"
MOTIVO_NAO_SNV = "nao_snv"
MOTIVO_FORA_DOS_AUTOSSOMOS = "fora_dos_autossomos"
MOTIVO_MEMBRO_DE_ESTUDO = "membro_de_estudo"
MOTIVO_CHR8 = "chr8_reservado"


def chave(chrom: str, pos: int, ref: str, alt: str) -> str:
    """Chave de alelo normalizada, comparavel entre o TSV (`1`) e o snapshot (`chr1`)."""
    return f"{normalize_chrom(chrom)}:{int(pos)}:{str(ref).upper()}:{str(alt).upper()}"


def e_snv(ref: str, alt: str) -> bool:
    ref_u, alt_u = str(ref).upper(), str(alt).upper()
    return len(ref_u) == 1 and len(alt_u) == 1 and ref_u in BASES and alt_u in BASES and ref_u != alt_u


def classificar(
    linhas: pd.DataFrame, *, membros: set[str], reservar_chr8: bool
) -> pd.Series:
    """Motivo de cada linha, na ordem em que as regras se aplicam."""
    chrom_norm = linhas["chrom"].map(normalize_chrom)
    motivos = pd.Series(MOTIVO_OK, index=linhas.index, dtype="object")

    eh_snv = pd.Series([e_snv(r, a) for r, a in zip(linhas["ref"], linhas["alt"])], index=linhas.index)
    motivos[~eh_snv] = MOTIVO_NAO_SNV

    fora = (~chrom_norm.isin(AUTOSSOMOS)) & (motivos == MOTIVO_OK)
    motivos[fora] = MOTIVO_FORA_DOS_AUTOSSOMOS

    if membros:
        chaves = pd.Series(
            [chave(c, p, r, a) for c, p, r, a in
             zip(linhas["chrom"], linhas["pos"], linhas["ref"], linhas["alt"])],
            index=linhas.index)
        membro = chaves.isin(membros) & (motivos == MOTIVO_OK)
        motivos[membro] = MOTIVO_MEMBRO_DE_ESTUDO

    if reservar_chr8:
        oito = (chrom_norm == "chr8") & (motivos == MOTIVO_OK)
        motivos[oito] = MOTIVO_CHR8
    return motivos


def distribuicao_af(af: pd.Series) -> dict[str, int]:
    cortes = pd.cut(af, bins=list(AF_BINS), include_lowest=True, right=True)
    return {str(intervalo): int(quantidade) for intervalo, quantidade in cortes.value_counts().sort_index().items()}


def chaves_do_parquet(caminho: Path, papel: str | None = None) -> set[str]:
    colunas = ["chrom", "pos_1based", "ref", "alt"]
    frame = pd.read_parquet(caminho, columns=colunas + (["role"] if papel else []))
    if papel:
        frame = frame[frame["role"] == papel]
    return {chave(c, p, r, a) for c, p, r, a in
            zip(frame["chrom"], frame["pos_1based"], frame["ref"], frame["alt"])}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--abraom", required=True, type=Path)
    parser.add_argument("--brazil-variants", type=Path, help="saida do G1: membros dos dois estudos")
    parser.add_argument("--snapshot", action="append", type=Path, default=None,
                        help="pode repetir; mede a sobreposicao com o treino de cada candidato")
    parser.add_argument("--no-reserve-chr8", action="store_true")
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args(argv)

    caminho = args.abraom.expanduser()
    linhas = pd.read_csv(caminho, sep="\t", dtype={"chrom": str, "ref": str, "alt": str})
    faltando = [c for c in COLUNAS if c not in linhas.columns]
    if faltando:
        print(f"FALHOU: faltam colunas {faltando} em {caminho}; achei {list(linhas.columns)}")
        return 2

    membros: set[str] = set()
    if args.brazil_variants:
        membros = chaves_do_parquet(args.brazil_variants.expanduser())

    motivos = classificar(linhas, membros=membros, reservar_chr8=not args.no_reserve_chr8)
    pool = linhas[motivos == MOTIVO_OK].copy()
    pool["chrom"] = pool["chrom"].map(normalize_chrom)

    sobreposicao_treino = {}
    for caminho_snapshot in (args.snapshot or []):
        treino = chaves_do_parquet(caminho_snapshot.expanduser(), papel="train")
        chaves_pool = {chave(c, p, r, a) for c, p, r, a in
                       zip(pool["chrom"], pool["pos"], pool["ref"], pool["alt"])}
        sobreposicao_treino[str(caminho_snapshot)] = len(chaves_pool & treino)

    out_dir = args.out_dir.expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    pool_path = out_dir / "abraom_pool.parquet"
    pool.to_parquet(pool_path, index=False)

    relatorio: dict[str, Any] = {
        "entrada": {"arquivo": str(caminho), "sha256": sha256_file(caminho), "linhas": int(len(linhas))},
        "convencao_de_cromossomo": {
            "no_arquivo": sorted(linhas["chrom"].astype(str).unique())[:5],
            "normalizado": "chrN, como o snapshot e o FASTA",
            "nota": "juntar sem normalizar daria zero sobreposicao em silencio",
        },
        "por_motivo": {motivo: int(quantidade) for motivo, quantidade in motivos.value_counts().items()},
        "duplicadas": int(len(linhas) - len(linhas.drop_duplicates(subset=["chrom", "pos", "ref", "alt"]))),
        "pool": {
            "n": int(len(pool)),
            "por_cromossomo": {str(c): int(n) for c, n in pool["chrom"].value_counts().sort_index().items()},
            "af": {
                "min": float(pool["af_abraom"].min()), "max": float(pool["af_abraom"].max()),
                "mediana": float(pool["af_abraom"].median()),
                "por_bin": distribuicao_af(pool["af_abraom"]),
            },
        },
        "sobreposicao_com_o_treino": sobreposicao_treino,
        "o_que_nao_prova": [
            "1,37 milhao de variantes e pouco para 1.171 genomas completos: o 'clean' e subconjunto curado",
            "a distribuicao por cromossomo nao e proporcional ao tamanho, entao a amostragem herda esse vies",
            "chr8 aqui e custo medido, nao decisao: a decisao E e do Eduardo",
        ],
        "saidas": {"pool": str(pool_path), "pool_sha256": sha256_file(pool_path)},
    }
    (out_dir / "auditoria_abraom.json").write_text(
        json.dumps(relatorio, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({k: relatorio[k] for k in
                      ("por_motivo", "duplicadas", "sobreposicao_com_o_treino", "saidas")},
                     ensure_ascii=False, indent=2))
    print(json.dumps({"pool_n": relatorio["pool"]["n"], "af": relatorio["pool"]["af"]},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
