#!/usr/bin/env python3
"""Audita o ABraOM do source-lock e publica o pool de amostragem do adapter, com identidade.

Roda no notebook (pandas + pyarrow + pyyaml; sem GPU). So le; escreve em --out-dir.
Plano: docs/proposta_mosaic_regionalizacao_desenvolvimento.md, secoes 4.3 e 5.1 (gate G4).

POR QUE EXISTE
--------------
Conferir sha256 prova QUAL arquivo e; nao prova o que ele contem nem o que pode ser usado. Antes de gerar janelas
sinteticas com ele:

1. **Convencao de cromossomo.** O TSV usa `1`; o snapshot e o FASTA usam `chr1`. Juntar sem normalizar daria
   zero sobreposicao em silencio -- o pior erro possivel, porque parece "nenhum vazamento".
2. **O que nao vira janela.** AF invalida ou degenerada, linha nao-SNV, fora de chr1..chr22: contadas e
   descartadas explicitamente, nunca encontradas no meio do treino.
3. **Alelo que sera pontuado depois.** Treinar o adapter a reconstruir exatamente o alelo de uma variante que
   sera avaliada e diferente de so ver o contexto genomico dela. Entao saem do pool os membros dos dois estudos,
   as variantes do conjunto de selecao e as da validacao e do teste do core. O treino da cabeca NAO sai: a
   sobreposicao com ele e medida e declarada, nao eliminada.

O QUE NAO PROVA
---------------
- Nao explica COMO o arquivo foi filtrado. 1,37 milhao de variantes para 1.171 genomas completos e compativel
  com filtragem, e o nome `clean` sugere isso, mas quais filtros foram aplicados so a documentacao da origem diz.
  Ate la, a amostragem herda o vies que houver, seja ele qual for.
- Nao decide chr8: mede o custo de reserva-lo. A decisao E continua do Eduardo.
- Nao confere REF contra o FASTA: isso e do gerador de janelas, e tem de acontecer antes do treino.

USO (notebook)
--------------
    export WORK=~/testeArq/lumina-beat-regionalization
    PYTHONPATH="$WORK" python3 "$WORK"/scripts/audit_abraom_source.py \\
        --abraom ~/artifacts/redesenho/g0_fontes/SABE1171.Abraom.clean.tsv \\
        --mosaic-root ~/testeArq/lumina-mosaic \\
        --brazil-variants ~/artifacts/redesenho/g1_brazil_studies/brazil_study_variants.parquet \\
        --selection ~/artifacts/redesenho/g5_comum/selecao_comum.parquet \\
        --snapshot ~/artifacts/redesenho/g2_final_nenhum/core_head_snapshot.parquet \\
        --out-dir ~/artifacts/redesenho/g4_abraom
    echo "exit=$?"
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_core_locus_head_snapshot import ROLE_TEST, ROLE_TRAIN, ROLE_VALIDATION, normalize_chrom
from scripts.import_mosaic_brazil_studies import sha256_file  # noqa: E402
from scripts.locate_abraom_source import expected_source  # noqa: E402

COLUNAS = ("chrom", "pos", "ref", "alt", "af_abraom")
BASES = frozenset("ACGT")

#: Autossomos: a suite do Mosaic e so chr1..chr22 (`mosaic.reference.PRIMARY_CHROMS`).
AUTOSSOMOS = tuple(f"chr{i}" for i in range(1, 23))

#: Bins de AF declarados, usados na amostragem estratificada da secao 5.1.
AF_BINS = (0.0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0)

MOTIVO_OK = "ok"
MOTIVO_AF_INVALIDA = "af_invalida"
MOTIVO_AF_DEGENERADA = "af_degenerada"
MOTIVO_NAO_SNV = "nao_snv"
MOTIVO_FORA_DOS_AUTOSSOMOS = "fora_dos_autossomos"
MOTIVO_MEMBRO_DE_ESTUDO = "membro_de_estudo"
MOTIVO_ALELO_DE_AVALIACAO = "alelo_de_avaliacao"
MOTIVO_CHR8 = "chr8_reservado"

ORDEM_DOS_MOTIVOS = (
    MOTIVO_AF_INVALIDA, MOTIVO_AF_DEGENERADA, MOTIVO_NAO_SNV, MOTIVO_FORA_DOS_AUTOSSOMOS,
    MOTIVO_MEMBRO_DE_ESTUDO, MOTIVO_ALELO_DE_AVALIACAO, MOTIVO_CHR8,
)


def chave(chrom: Any, pos: Any, ref: Any, alt: Any) -> str:
    """Chave de alelo normalizada, comparavel entre o TSV (`1`) e o snapshot (`chr1`)."""
    return f"{normalize_chrom(chrom)}:{int(pos)}:{str(ref).upper()}:{str(alt).upper()}"


def e_snv(ref: Any, alt: Any) -> bool:
    ref_u, alt_u = str(ref).upper(), str(alt).upper()
    return len(ref_u) == 1 and len(alt_u) == 1 and ref_u in BASES and alt_u in BASES and ref_u != alt_u


def af_invalida(valor: Any) -> bool:
    try:
        numero = float(valor)
    except (TypeError, ValueError):
        return True
    return math.isnan(numero) or math.isinf(numero) or numero < 0.0 or numero > 1.0


def af_degenerada(valor: Any) -> bool:
    """AF 0 (nao observada na coorte) ou 1 (fixada) nao e variacao populacional utilizavel."""
    numero = float(valor)
    return numero <= 0.0 or numero >= 1.0


def classificar(
    linhas: pd.DataFrame,
    *,
    membros: set[str],
    alelos_de_avaliacao: set[str],
    reservar_chr8: bool,
    manter_af_degenerada: bool = False,
) -> pd.Series:
    """Motivo de cada linha, primeira regra que se aplica. A ordem esta em ORDEM_DOS_MOTIVOS."""
    motivos = pd.Series(MOTIVO_OK, index=linhas.index, dtype="object")
    chrom_norm = linhas["chrom"].map(normalize_chrom)
    chaves = pd.Series(
        [chave(c, p, r, a) for c, p, r, a in
         zip(linhas["chrom"], linhas["pos"], linhas["ref"], linhas["alt"])],
        index=linhas.index)

    def aplicar(mascara: pd.Series, motivo: str) -> None:
        motivos[mascara & (motivos == MOTIVO_OK)] = motivo

    aplicar(linhas["af_abraom"].map(af_invalida), MOTIVO_AF_INVALIDA)
    if not manter_af_degenerada:
        # So avalia onde o AF ja passou pela checagem de validade; `aplicar` cuida de nao sobrescrever motivo.
        degenerada = pd.Series(
            [motivo == MOTIVO_OK and af_degenerada(valor)
             for motivo, valor in zip(motivos, linhas["af_abraom"])],
            index=linhas.index, dtype=bool)
        aplicar(degenerada, MOTIVO_AF_DEGENERADA)
    aplicar(~pd.Series([e_snv(r, a) for r, a in zip(linhas["ref"], linhas["alt"])], index=linhas.index),
            MOTIVO_NAO_SNV)
    aplicar(~chrom_norm.isin(AUTOSSOMOS), MOTIVO_FORA_DOS_AUTOSSOMOS)
    if membros:
        aplicar(chaves.isin(membros), MOTIVO_MEMBRO_DE_ESTUDO)
    if alelos_de_avaliacao:
        aplicar(chaves.isin(alelos_de_avaliacao), MOTIVO_ALELO_DE_AVALIACAO)
    if reservar_chr8:
        aplicar(chrom_norm == "chr8", MOTIVO_CHR8)
    return motivos


def duplicatas_conflitantes(linhas: pd.DataFrame) -> pd.DataFrame:
    """Mesmo alelo com AF diferente: nao da para escolher em silencio qual vale."""
    chaves = [chave(c, p, r, a) for c, p, r, a in
              zip(linhas["chrom"], linhas["pos"], linhas["ref"], linhas["alt"])]
    frame = pd.DataFrame({"chave": chaves, "af": linhas["af_abraom"].to_numpy()})
    por_chave = frame.groupby("chave")["af"].nunique()
    conflitantes = por_chave[por_chave > 1]
    return frame[frame["chave"].isin(conflitantes.index)]


def distribuicao_af(af: pd.Series) -> dict[str, int]:
    cortes = pd.cut(af, bins=list(AF_BINS), include_lowest=True, right=True)
    return {str(intervalo): int(quantidade)
            for intervalo, quantidade in cortes.value_counts().sort_index().items()}


def chaves_do_parquet(caminho: Path, papeis: tuple[str, ...] | None = None) -> set[str]:
    colunas = ["chrom", "pos_1based", "ref", "alt"]
    disponiveis = set(pd.read_parquet(caminho).columns) if papeis else None
    if papeis and "role" in (disponiveis or set()):
        frame = pd.read_parquet(caminho, columns=colunas + ["role"])
        frame = frame[frame["role"].isin(papeis)]
    else:
        frame = pd.read_parquet(caminho, columns=colunas)
    return {chave(c, p, r, a) for c, p, r, a in
            zip(frame["chrom"], frame["pos_1based"], frame["ref"], frame["alt"])}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--abraom", required=True, type=Path)
    parser.add_argument("--mosaic-root", type=Path,
                        help="clone do Mosaic: confere o sha256 do ABraOM contra o source-lock")
    parser.add_argument("--brazil-variants", type=Path, help="saida do G1: membros dos dois estudos")
    parser.add_argument("--selection", type=Path, help="conjunto de selecao comum")
    parser.add_argument("--snapshot", action="append", type=Path, default=None,
                        help="pode repetir; validacao e teste saem do pool, treino e so medido")
    parser.add_argument("--no-reserve-chr8", action="store_true")
    parser.add_argument("--manter-af-degenerada", action="store_true",
                        help="mantem AF 0 e 1 no pool (por padrao saem, com motivo declarado)")
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args(argv)

    caminho = args.abraom.expanduser()
    linhas = pd.read_csv(caminho, sep="\t", dtype={"chrom": str, "ref": str, "alt": str})
    faltando = [c for c in COLUNAS if c not in linhas.columns]
    if faltando:
        print(f"FALHOU: faltam colunas {faltando} em {caminho}; achei {list(linhas.columns)}")
        return 2

    entrada: dict[str, Any] = {"arquivo": str(caminho), "sha256": sha256_file(caminho),
                               "linhas": int(len(linhas))}
    if args.mosaic_root:
        spec = expected_source(args.mosaic_root)
        entrada["sha256_declarado"] = spec["sha256"]
        entrada["bate_com_o_source_lock"] = entrada["sha256"] == spec["sha256"]
        if not entrada["bate_com_o_source_lock"]:
            print(f"FALHOU: sha256 {entrada['sha256']} difere do declarado {spec['sha256']}")
            return 2

    conflitantes = duplicatas_conflitantes(linhas)
    if len(conflitantes):
        print(f"FALHOU: {conflitantes['chave'].nunique()} alelos com AF conflitante; "
              f"ex.: {conflitantes.head(3).to_dict(orient='records')}")
        return 2

    identidades: dict[str, str] = {}
    membros: set[str] = set()
    if args.brazil_variants:
        membros = chaves_do_parquet(args.brazil_variants.expanduser())
        identidades[str(args.brazil_variants)] = sha256_file(args.brazil_variants.expanduser())

    avaliacao: set[str] = set()
    if args.selection:
        avaliacao |= chaves_do_parquet(args.selection.expanduser())
        identidades[str(args.selection)] = sha256_file(args.selection.expanduser())
    sobreposicao_treino: dict[str, int] = {}
    for caminho_snapshot in (args.snapshot or []):
        expandido = caminho_snapshot.expanduser()
        avaliacao |= chaves_do_parquet(expandido, papeis=(ROLE_VALIDATION, ROLE_TEST))
        identidades[str(caminho_snapshot)] = sha256_file(expandido)

    pode_publicar = bool(membros) and bool(avaliacao)
    motivos = classificar(linhas, membros=membros, alelos_de_avaliacao=avaliacao,
                          reservar_chr8=not args.no_reserve_chr8,
                          manter_af_degenerada=args.manter_af_degenerada)
    pool = linhas[motivos == MOTIVO_OK].copy()
    pool["chrom"] = pool["chrom"].map(normalize_chrom)

    chaves_pool = {chave(c, p, r, a) for c, p, r, a in
                   zip(pool["chrom"], pool["pos"], pool["ref"], pool["alt"])}
    for caminho_snapshot in (args.snapshot or []):
        treino = chaves_do_parquet(caminho_snapshot.expanduser(), papeis=(ROLE_TRAIN,))
        sobreposicao_treino[str(caminho_snapshot)] = len(chaves_pool & treino)

    out_dir = args.out_dir.expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    saidas: dict[str, Any] = {}
    if pode_publicar:
        pool_path = out_dir / "abraom_pool.parquet"
        pool.to_parquet(pool_path, index=False)
        saidas = {"pool": str(pool_path), "pool_sha256": sha256_file(pool_path)}

    relatorio: dict[str, Any] = {
        "entrada": entrada,
        "identidades_das_exclusoes": identidades,
        "convencao_de_cromossomo": {
            "no_arquivo": sorted(linhas["chrom"].astype(str).unique())[:5],
            "normalizado": "chrN, como o snapshot e o FASTA",
        },
        "por_motivo": {motivo: int(quantidade) for motivo, quantidade in motivos.value_counts().items()},
        "ordem_das_regras": list(ORDEM_DOS_MOTIVOS),
        "duplicatas_exatas": int(len(linhas) - len(linhas.drop_duplicates(subset=list(COLUNAS[:4])))),
        "pool": {
            "n": int(len(pool)),
            "por_cromossomo": {str(c): int(n) for c, n in pool["chrom"].value_counts().sort_index().items()},
            "af": {
                "min": float(pool["af_abraom"].min()) if len(pool) else None,
                "max": float(pool["af_abraom"].max()) if len(pool) else None,
                "mediana": float(pool["af_abraom"].median()) if len(pool) else None,
                "por_bin": distribuicao_af(pool["af_abraom"]) if len(pool) else {},
            },
        },
        "sobreposicao_com_o_treino_da_cabeca": sobreposicao_treino,
        "pronto_para_treino": pode_publicar,
        "pendencias": [] if pode_publicar else [
            "faltou --brazil-variants e/ou os recortes de avaliacao: o pool NAO foi publicado"
        ],
        "o_que_nao_prova": [
            "nao explica como o arquivo foi filtrado: o tamanho e compativel com filtragem, os filtros em si "
            "so a documentacao da origem diz",
            "sobreposicao com o treino da cabeca e declarada, nao eliminada: ver contexto nao e o mesmo que "
            "treinar para reconstruir o alelo que sera pontuado",
            "chr8 aqui e custo medido, nao decisao",
        ],
        "saidas": saidas,
    }
    (out_dir / "auditoria_abraom.json").write_text(
        json.dumps(relatorio, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({k: relatorio[k] for k in
                      ("por_motivo", "duplicatas_exatas", "sobreposicao_com_o_treino_da_cabeca",
                       "pronto_para_treino", "pendencias", "saidas")}, ensure_ascii=False, indent=2))
    print(json.dumps({"pool_n": relatorio["pool"]["n"], "af": relatorio["pool"]["af"]},
                     ensure_ascii=False, indent=2))
    return 0 if pode_publicar else 2


if __name__ == "__main__":
    sys.exit(main())
