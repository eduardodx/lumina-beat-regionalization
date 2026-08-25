#!/usr/bin/env python3
"""Anota as variantes de TP53 das amostras com a classificacao do ClinVar (o valor clinico).

Cruza cada variante chamada (Clair3/Nanopore) com o ClinVar na regiao de TP53 e reporta a
significancia clinica (CLNSIG) + doenca (CLNDN). Consolida as 6 amostras numa tabela e DESTACA
qualquer variante Pathogenic / Likely_pathogenic — que e o que interessa reportar ao Eduardo.

Match por chave canonica (chrom sem 'chr', pos, ref, alt); rode `bcftools norm` nos VCFs antes p/
alinhar a representacao de indels com a do ClinVar. Naming de contig do ClinVar (17 vs chr17) e
normalizado automaticamente.

Roda no env com pysam (`clair3`):
    conda run -n clair3 python seq_pipeline/annotate_tp53_clinvar.py \
        --clinvar ~/seqlab/clinvar/clinvar_20260606.vcf.gz --vcf-dir ~/seqlab/vcf \
        --samples 353 358 361 362 370 674 --out ~/seqlab/vcf/tp53_clinvar_annotation.tsv
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import pysam

TP53 = ("17", 7_660_000, 7_695_000)  # GRCh38, janela do amplicon
PLP = ("pathogenic", "likely_pathogenic")


def norm_chrom(c: str) -> str:
    return c[3:] if c.lower().startswith("chr") else c


def fmt(v) -> str:
    if v is None:
        return "."
    if isinstance(v, (tuple, list)):
        return "|".join(str(x) for x in v)
    return str(v)


def load_clinvar(path: str) -> dict[tuple, tuple[str, str]]:
    vf = pysam.VariantFile(path)
    contigs = set(vf.header.contigs)
    chrom = TP53[0] if TP53[0] in contigs else ("chr" + TP53[0] if "chr" + TP53[0] in contigs else TP53[0])
    out: dict[tuple, tuple[str, str]] = {}
    try:
        it = vf.fetch(chrom, TP53[1], TP53[2])
    except (ValueError, OSError) as exc:
        sys.exit(f"Nao consegui fetch do ClinVar em {chrom}:{TP53[1]}-{TP53[2]} ({exc}). "
                 "O ClinVar precisa estar indexado (bcftools index -t).")
    for rec in it:
        clnsig = fmt(rec.info.get("CLNSIG"))
        clndn = fmt(rec.info.get("CLNDN"))
        for alt in (rec.alts or []):
            out[(norm_chrom(rec.chrom), rec.pos, rec.ref, alt)] = (clnsig, clndn)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clinvar", required=True, help="ClinVar VCF indexado (.tbi)")
    ap.add_argument("--vcf-dir", type=Path, required=True)
    ap.add_argument("--samples", nargs="+", required=True)
    ap.add_argument("--vcf-glob", default="{sample}_ont_tp53/merge_output.vcf.gz")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    clinvar = load_clinvar(args.clinvar)
    print(f"# ClinVar em TP53: {len(clinvar)} alelos classificados", file=sys.stderr)

    # variante -> {sample: (ont_af, filter)} e a classificacao ClinVar
    hits: dict[tuple, dict] = defaultdict(lambda: {"samples": {}, "clnsig": ".", "clndn": "."})
    for s in args.samples:
        vpath = args.vcf_dir / args.vcf_glob.format(sample=s)
        if not vpath.is_file():
            print(f"# aviso: {vpath} nao encontrado", file=sys.stderr)
            continue
        for rec in pysam.VariantFile(str(vpath)):
            for alt in (rec.alts or []):
                key = (norm_chrom(rec.chrom), rec.pos, rec.ref, alt)
                cv = clinvar.get(key)
                if cv is None:
                    continue
                af = rec.samples[0].get("AF")
                af = af[0] if isinstance(af, (tuple, list)) else af
                filt = ";".join(rec.filter.keys()) or "PASS"
                hits[key]["samples"][s] = (f"{float(af):.2f}" if af is not None else "?", filt)
                hits[key]["clnsig"], hits[key]["clndn"] = cv

    rows = sorted(hits.items(), key=lambda kv: (kv[0][0], kv[0][1]))
    header = ["variante", "clnsig", "clndn", "n_amostras", "amostras(AF)"]
    print("\t".join(header))
    plp_rows = []
    for (chrom, pos, ref, alt), info in rows:
        variante = f"chr{chrom}:{pos} {ref}>{alt}"
        samps = ",".join(f"{s}({af},{fl})" for s, (af, fl) in info["samples"].items())
        line = [variante, info["clnsig"], info["clndn"], str(len(info["samples"])), samps]
        print("\t".join(line))
        if any(p in info["clnsig"].lower() for p in PLP):
            plp_rows.append(line)

    print(f"\n# {len(rows)} variantes das amostras têm classificação ClinVar em TP53.")
    if plp_rows:
        print(f"# ⚠ {len(plp_rows)} PATHOGENIC/LIKELY_PATHOGENIC encontradas:")
        for line in plp_rows:
            print("#   " + " | ".join(line[:4]))
    else:
        print("# Nenhuma Pathogenic/Likely_pathogenic — as classificadas são benignas/VUS "
              "(consistente com polimorfismos germinativos).")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write("\t".join(header) + "\n")
            for (chrom, pos, ref, alt), info in rows:
                samps = ",".join(f"{s}({af},{fl})" for s, (af, fl) in info["samples"].items())
                fh.write("\t".join([f"chr{chrom}:{pos} {ref}>{alt}", info["clnsig"], info["clndn"],
                                    str(len(info["samples"])), samps]) + "\n")
        print(f"# salvo: {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
