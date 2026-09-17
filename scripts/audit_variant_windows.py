#!/usr/bin/env python3
"""Audita as janelas de sequencia da campanha: REF contra o FASTA, bordas e bases fora de ACGT.

Roda no notebook (pandas + pyarrow + pysam/pyfaidx; sem GPU). So le; escreve em --out-dir.
Plano: docs/proposta_mosaic_regionalizacao_desenvolvimento.md, secao 5.2.

POR QUE EXISTE
--------------
Antes de gastar GPU com extracao, e preciso saber quantas variantes do conjunto sequer produzem janela valida, e
por que. Sao tres motivos, com politicas diferentes:

  `ref_mismatch`  a base do FASTA na posicao focal nao e o REF declarado. O release do Mosaic conferiu o REF
                  contra o GRCh38.p14, entao isto significa FASTA ou build errado -- e ERRO, nao estatistica:
                  o script sai com codigo 2.
  `non_acgt`      a janela tem base fora de ACGT (N). A politica declarada e DESCARTAR a variante; o soft-mask
                  (minusculas) e normalizado antes, entao nao conta aqui.
  `out_of_bounds` a janela nao cabe no cromossomo. Descartada tambem -- deslocar tiraria a variante do indice
                  focal declarado.

O numero que interessa para a declaracao e quantas variantes cada motivo tira, por papel e por painel.

COMO
----
Usa `eval/embedding_probe/windows.py` (portado da pesquisa), que implementa a convencao do Mosaic: offset focal
`L // 2 - 1`, validacao estrita sem fallback e nenhuma janela deslocada.

O QUE NAO PROVA
---------------
- Nao valida o conteudo biologico da janela, so o contrato: REF confere, cabe, e e ACGT.
- Nao decide o tamanho da janela: ele e declarado (`--window-bp`).

USO (notebook)
--------------
    export WORK=~/testeArq/lumina-beat-regionalization
    PYTHONPATH="$WORK" python3 "$WORK"/scripts/audit_variant_windows.py \
        --variants ~/artifacts/redesenho/g2_final_janela4096/core_head_snapshot.parquet \
        --fasta ~/hg38/hg38.fa --window-bp 4096 \
        --out-dir ~/artifacts/redesenho/g3_janelas/treino4096
    echo "exit=$?"
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.embedding_probe.windows import WindowError, build_window, focal_offset  # noqa: E402

COLUNAS = ("variant_id", "chrom", "pos_1based", "ref", "alt")


def audit_windows(
    rows: pd.DataFrame, fetch: Callable[[str, int, int], str], *, window_bp: int
) -> tuple[Counter, list[dict[str, Any]]]:
    """Constroi a janela de cada linha e agrega por motivo. Nao levanta: coleta."""
    motivos: Counter = Counter()
    falhas: list[dict[str, Any]] = []
    for row in rows.itertuples(index=False):
        try:
            build_window(fetch, chrom=str(row.chrom), pos_1based=int(row.pos_1based),
                         ref=str(row.ref), alt=str(row.alt), window_bp=window_bp)
            motivos["ok"] += 1
        except WindowError as exc:
            motivos[exc.reason] += 1
            falhas.append({
                "variant_id": str(row.variant_id), "chrom": str(row.chrom),
                "pos_1based": int(row.pos_1based), "ref": str(row.ref), "alt": str(row.alt),
                "motivo": exc.reason, "detalhe": exc.detail,
                **({"role": row.role} if hasattr(row, "role") else {}),
                **({"primary_panel": row.primary_panel} if hasattr(row, "primary_panel") else {}),
                **({"binary_label": int(row.binary_label)} if hasattr(row, "binary_label") else {}),
            })
    return motivos, falhas


def breakdown(falhas: list[dict[str, Any]], coluna: str) -> dict[str, dict[str, int]]:
    """Falhas por motivo e por uma coluna de contexto (papel, painel, classe)."""
    out: dict[str, dict[str, int]] = {}
    for falha in falhas:
        if coluna not in falha:
            continue
        out.setdefault(falha["motivo"], {})
        chave = str(falha[coluna])
        out[falha["motivo"]][chave] = out[falha["motivo"]].get(chave, 0) + 1
    return out


def abrir_fasta(caminho: Path):
    """Devolve um `fetch(chrom, start0, end0)`; aceita pysam ou pyfaidx, nesta ordem."""
    try:
        import pysam

        handle = pysam.FastaFile(str(caminho))
        return lambda chrom, start, end: handle.fetch(chrom, start, end), "pysam"
    except ImportError:
        pass
    try:
        from pyfaidx import Fasta

        handle = Fasta(str(caminho), as_raw=True, sequence_always_upper=False)
        return lambda chrom, start, end: str(handle[chrom][start:end]), "pyfaidx"
    except ImportError as exc:
        raise SystemExit(f"nem pysam nem pyfaidx disponiveis para ler {caminho}: {exc}") from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--variants", required=True, type=Path)
    parser.add_argument("--fasta", required=True, type=Path)
    parser.add_argument("--window-bp", type=int, default=4096)
    parser.add_argument("--limit", type=int, help="audita so as N primeiras linhas (sondagem rapida)")
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args(argv)

    rows = pd.read_parquet(args.variants.expanduser())
    faltando = [c for c in COLUNAS if c not in rows.columns]
    if faltando:
        print(f"FALHOU: faltam colunas {faltando} em {args.variants}")
        return 2
    if args.limit:
        rows = rows.head(args.limit)

    fetch, leitor = abrir_fasta(args.fasta.expanduser())
    motivos, falhas = audit_windows(rows, fetch, window_bp=args.window_bp)

    out_dir = args.out_dir.expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    if falhas:
        pd.DataFrame(falhas).to_parquet(out_dir / "variantes_sem_janela.parquet", index=False)

    total = int(len(rows))
    relatorio: dict[str, Any] = {
        "entradas": {"variantes": str(args.variants), "fasta": str(args.fasta), "leitor": leitor,
                     "window_bp": args.window_bp, "focal_index": focal_offset(args.window_bp)},
        "total": total,
        "por_motivo": dict(sorted(motivos.items())),
        "fracao_descartada": round((total - motivos["ok"]) / total, 6) if total else None,
        "por_papel": breakdown(falhas, "role"),
        "por_painel": breakdown(falhas, "primary_panel"),
        "por_classe": breakdown(falhas, "binary_label"),
        "politica": {
            "non_acgt": "variante descartada (soft-mask normalizado antes; N nao e substituido)",
            "out_of_bounds": "variante descartada; janela nunca e deslocada",
            "ref_mismatch": "ERRO: o release conferiu REF contra GRCh38.p14, entao isto e FASTA ou build errado",
        },
        "o_que_nao_prova": ["nao valida o conteudo biologico da janela, so o contrato"],
    }
    (out_dir / "auditoria_de_janelas.json").write_text(
        json.dumps(relatorio, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({k: relatorio[k] for k in ("total", "por_motivo", "fracao_descartada", "por_painel")},
                     ensure_ascii=False, indent=2))

    if motivos.get("ref_mismatch"):
        print(f"\nFALHOU: {motivos['ref_mismatch']} variantes com ref_mismatch -- FASTA ou build errado.")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
