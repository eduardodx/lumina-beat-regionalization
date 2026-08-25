#!/usr/bin/env python3
"""Passo 3 do Eduardo: para cada variante do Nanopore (TP53), verifica o suporte no Ion.

Em vez de chamar o Ion independentemente (caller instavel + o painel Ion so cobre os EXONS de TP53, nao
o gene todo que o amplicon Nanopore cobre), este script faz **genotipagem direcionada**: para cada
variante chamada no Nanopore, conta no BAM do Ion, naquela posicao exata, quantos reads suportam o
alelo alternativo. Isso separa tres casos e responde direto "a variante do Nanopore esta no Ion?":

  * NAO_COBERTA   -> Ion tem profundidade < --min-dp na posicao (fora do amplicon Ion; nao avaliavel)
  * CONFIRMADA    -> Ion cobre e ve o alt (Ion_AF >= --min-alt-af)
  * NAO_CONFIRMADA-> Ion cobre bem mas nao ve o alt (candidata a artefato do Nanopore, ex.: homopolimero)

Roda no env com pysam (o `clair3`):
    conda run -n clair3 python seq_pipeline/compare_tp53_ont_vs_ion.py \
        --ont-vcf ~/seqlab/vcf/674_ont_tp53/merge_output.vcf.gz \
        --ion-bam ~/seqlab/aln/674_ion.bam --sample 674 --out ~/seqlab/vcf/674_concordance.tsv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pysam


def variant_kind(ref: str, alt: str) -> str:
    if len(ref) == 1 and len(alt) == 1:
        return "SNV"
    return "INDEL"


def ion_support(bam: pysam.AlignmentFile, chrom: str, pos1: int, ref: str, alt: str,
                max_depth: int = 200000) -> tuple[int, int]:
    """Retorna (profundidade, reads_suportando_alt) no BAM do Ion na posicao pos1 (1-based)."""
    pos0 = pos1 - 1
    depth = 0
    alt_support = 0
    is_snv = len(ref) == 1 and len(alt) == 1
    ins_len = len(alt) - len(ref)  # >0 insercao, <0 delecao, 0 SNV/MNV
    for col in bam.pileup(chrom, pos0, pos0 + 1, truncate=True, min_base_quality=0,
                          max_depth=max_depth, stepper="samtools"):
        if col.reference_pos != pos0:
            continue
        for pr in col.pileups:
            depth += 1
            if is_snv:
                if pr.is_del or pr.is_refskip or pr.query_position is None:
                    continue
                base = pr.alignment.query_sequence[pr.query_position]
                if base.upper() == alt.upper():
                    alt_support += 1
            else:
                # indel aparece em pr.indel na coluna ANCORA (a base antes do evento)
                if ins_len > 0 and pr.indel == ins_len:
                    alt_support += 1
                elif ins_len < 0 and pr.indel == ins_len:
                    alt_support += 1
    return depth, alt_support


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ont-vcf", type=Path, required=True, help="merge_output.vcf.gz do Clair3/Nanopore")
    ap.add_argument("--ion-bam", type=Path, required=True)
    ap.add_argument("--sample", default="?")
    ap.add_argument("--out", type=Path, default=None, help="TSV de saida")
    ap.add_argument("--min-dp", type=int, default=10, help="profundidade minima do Ion p/ 'avaliavel'")
    ap.add_argument("--min-alt-af", type=float, default=0.10, help="Ion_AF minima p/ CONFIRMADA")
    ap.add_argument("--pass-only", action="store_true", help="so variantes PASS do Nanopore")
    args = ap.parse_args(argv)

    vcf = pysam.VariantFile(str(args.ont_vcf))
    bam = pysam.AlignmentFile(str(args.ion_bam))

    rows = []
    counts = {"CONFIRMADA": 0, "NAO_CONFIRMADA": 0, "NAO_COBERTA": 0}
    for rec in vcf:
        filt = set(rec.filter.keys())
        is_pass = ("PASS" in filt) or (not filt)
        if args.pass_only and not is_pass:
            continue
        ref, alt = rec.ref, rec.alts[0] if rec.alts else "."
        kind = variant_kind(ref, alt)
        try:
            ont_af = float(rec.samples[args.sample if args.sample in rec.samples else 0].get("AF", [None])[0]
                           if isinstance(rec.samples[0].get("AF"), (list, tuple)) else rec.samples[0].get("AF"))
        except Exception:  # noqa: BLE001
            ont_af = float("nan")
        dp, altc = ion_support(bam, rec.chrom, rec.pos, ref, alt)
        ion_af = (altc / dp) if dp > 0 else 0.0
        if dp < args.min_dp:
            verdict = "NAO_COBERTA"
        elif ion_af >= args.min_alt_af:
            verdict = "CONFIRMADA"
        else:
            verdict = "NAO_CONFIRMADA"
        counts[verdict] += 1
        rows.append((f"{rec.chrom}:{rec.pos}", f"{ref}>{alt}", kind, is_pass,
                     f"{ont_af:.3f}" if ont_af == ont_af else "?", dp, altc, f"{ion_af:.3f}", verdict))

    header = ["posicao", "ref>alt", "tipo", "ont_PASS", "ONT_AF", "Ion_DP", "Ion_alt", "Ion_AF", "veredito"]
    print(f"# amostra {args.sample} | TP53 | Nanopore->Ion (min_dp={args.min_dp}, min_alt_af={args.min_alt_af})")
    print("\t".join(header))
    for r in rows:
        print("\t".join(str(x) for x in r))
    print(f"# RESUMO: CONFIRMADA={counts['CONFIRMADA']} "
          f"NAO_CONFIRMADA={counts['NAO_CONFIRMADA']} NAO_COBERTA={counts['NAO_COBERTA']} "
          f"(total={sum(counts.values())})")
    # so SNVs (a leitura confiavel; indels sofrem homopolimero nas duas plataformas)
    snv = [r for r in rows if r[2] == "SNV"]
    snv_conf = sum(1 for r in snv if r[8] == "CONFIRMADA")
    snv_evaluable = sum(1 for r in snv if r[8] != "NAO_COBERTA")
    if snv_evaluable:
        print(f"# SNVs avaliaveis (Ion cobre): {snv_conf}/{snv_evaluable} confirmadas "
              f"({100*snv_conf/snv_evaluable:.0f}%)")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write("\t".join(header) + "\n")
            for r in rows:
                fh.write("\t".join(str(x) for x in r) + "\n")
        print(f"# salvo: {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
