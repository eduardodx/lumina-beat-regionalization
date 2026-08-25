#!/usr/bin/env python3
"""Anotacao FUNCIONAL das variantes de TP53 via VEP REST (Ensembl) — fecha o gap da anotacao so-ClinVar.

O `annotate_tp53_clinvar.py` so acha variantes JA catalogadas no ClinVar. Uma variante patogenica NOVA
(ex.: stop_gained / frameshift / splice em TP53 nao presente no ClinVar) escaparia. Este script pega a
CONSEQUENCIA funcional de cada variante e sinaliza as que merecem atencao mesmo sem estar no ClinVar:

  * ALTO_IMPACTO  -> impacto HIGH (stop_gained, frameshift, splice_donor/acceptor) — candidata patogenica
  * MISSENSE_RARO -> missense com gnomAD AF ausente/baixa (<0.001) — merece revisao
  * CLINVAR_P/LP  -> o proprio VEP traz clin_sig pathogenic/likely_pathogenic (nao 'conflicting')
  * ok            -> sinonima/intronica/UTR ou missense comum (polimorfismo)

Usa o VEP REST publico (precisa de internet no notebook). Le so variantes PASS. Roda no env `clair3`
(pysam + urllib stdlib):
    conda run -n clair3 python seq_pipeline/annotate_tp53_vep.py \
        --vcf-dir ~/seqlab/vcf --samples 353 358 361 362 370 674 --out ~/seqlab/vcf/tp53_vep.tsv
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path

import pysam

VEP_URL = "https://rest.ensembl.org/vep/human/region"
IMPACT_RANK = {"HIGH": 3, "MODERATE": 2, "LOW": 1, "MODIFIER": 0}


def norm_chrom(c: str) -> str:
    return c[3:] if c.lower().startswith("chr") else c


def collect(vcf_dir: Path, samples, glob: str, pass_only: bool) -> dict[tuple, list]:
    var2s: dict[tuple, list] = defaultdict(list)
    for s in samples:
        p = vcf_dir / glob.format(sample=s)
        if not p.is_file():
            print(f"# aviso: {p} nao encontrado", file=sys.stderr)
            continue
        for rec in pysam.VariantFile(str(p)):
            filt = set(rec.filter.keys())
            if pass_only and filt and "PASS" not in filt:
                continue
            for alt in (rec.alts or []):
                var2s[(norm_chrom(rec.chrom), rec.pos, rec.ref, alt)].append(s)
    return var2s


def vep(variants: list[tuple]) -> dict[str, dict]:
    payload = json.dumps({"variants": [f"{c} {p} . {r} {a}" for (c, p, r, a) in variants]}).encode()
    req = urllib.request.Request(
        VEP_URL, data=payload,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            data = json.loads(resp.read())
    except (urllib.error.URLError, TimeoutError) as exc:
        sys.exit(f"ERRO no VEP REST ({exc}). Se o notebook nao tem internet publica, use SnpEff offline "
                 "(conda create -n snpeff -c bioconda snpeff; snpEff download GRCh38.105; snpEff GRCh38.105 in.vcf).")
    out = {}
    for item in data:
        out[item.get("input", "").strip()] = item
    return out


def parse(item: dict) -> tuple[str, str, float | None, list[str]]:
    csq = item.get("most_severe_consequence", "?")
    impact = "MODIFIER"
    for tc in item.get("transcript_consequences", []):
        if tc.get("gene_symbol") in (None, "TP53") and IMPACT_RANK.get(tc.get("impact", "MODIFIER"), 0) >= IMPACT_RANK.get(impact, 0):
            impact = tc.get("impact", "MODIFIER")
    gnomad = None
    clin: list[str] = []
    for cv in item.get("colocated_variants", []):
        clin += cv.get("clin_sig", []) or []
        for _allele, fr in (cv.get("frequencies", {}) or {}).items():
            for key in ("gnomade", "gnomadg", "gnomad"):
                if key in fr:
                    gnomad = fr[key] if gnomad is None else max(gnomad, fr[key])
    return csq, impact, gnomad, sorted(set(clin))


def verdict(impact: str, gnomad: float | None, clin: list[str]) -> str:
    """Veredito por variante. CHAVE: gnomAD AF >= 1% => polimorfismo comum, NUNCA candidata patogenica
    (variantes patogenicas de TP53/Li-Fraumeni sao raras) — descarta HIGH-impact e 'pathogenic' espurios
    em homopolimero/borda de gene e o proprio rs1042522 (P72R, AF ~72%)."""
    cl = " ".join(clin).lower()
    if gnomad is not None and gnomad >= 0.01:
        return "ok"  # comum na populacao -> benigno, independe de impacto/clin_sig
    if ("pathogenic" in cl) and ("conflicting" not in cl) and ("benign" not in cl):
        return "⚠ CLINVAR_P/LP"
    if impact == "HIGH":
        return "⚠ ALTO_IMPACTO"
    if impact == "MODERATE" and (gnomad is None or gnomad < 0.001) and ("benign" not in cl):
        return "⚠ MISSENSE_RARO"
    if ("uncertain" in cl or "conflicting" in cl) and "pathogenic" not in cl:
        return "~ VUS/revisar"
    return "ok"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vcf-dir", type=Path, required=True)
    ap.add_argument("--samples", nargs="+", required=True)
    ap.add_argument("--vcf-glob", default="{sample}_ont_tp53/merge_output.vcf.gz")
    ap.add_argument("--all-variants", action="store_true", help="inclui nao-PASS (default: so PASS)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    var2s = collect(args.vcf_dir, args.samples, args.vcf_glob, not args.all_variants)
    variants = sorted(var2s)
    print(f"# {len(variants)} variantes unicas (PASS) enviadas ao VEP", file=sys.stderr)
    ann = vep(variants)

    rows = []
    attention = []
    for v in variants:
        c, p, r, a = v
        item = ann.get(f"{c} {p} . {r} {a}")
        if item is None:
            csq, impact, gnomad, clin, vd = "?", "?", None, [], "sem_resposta_VEP"
        else:
            csq, impact, gnomad, clin = parse(item)
            vd = verdict(impact, gnomad, clin)
        row = [f"chr{c}:{p} {r}>{a}", csq, impact, "." if gnomad is None else f"{gnomad:.4g}",
               "|".join(clin) or ".", ",".join(var2s[v]), vd]
        rows.append(row)
        if vd != "ok":
            attention.append(row)

    header = ["variante", "consequencia", "impacto", "gnomAD_AF", "clin_sig", "amostras", "veredito"]
    print("\t".join(header))
    for row in sorted(rows, key=lambda r: (r[6] == "ok", r[0])):
        print("\t".join(row))

    candidatas = [r for r in rows if r[6].startswith("⚠")]
    vus = [r for r in rows if r[6].startswith("~")]
    print(f"\n# {len(variants)} variantes; candidatas patogênicas={len(candidatas)}; VUS/revisar={len(vus)}")
    if candidatas:
        print("# ⚠⚠ REVISAR (rara + alto impacto / P-LP):")
        for r in candidatas:
            print("#   " + " | ".join([r[0], r[1], r[2], f"gnomAD={r[3]}", r[5]]))
    else:
        print("# ✓ Nenhuma candidata patogênica (0 P/LP, 0 alto-impacto raro). '0 patogênicas de TP53' "
              "confirmado por 3 eixos: consequência funcional + frequência populacional (gnomAD) + ClinVar.")
    for r in vus:
        print("# ~ VUS/revisão manual: " + " | ".join([r[0], r[1], f"gnomAD={r[3]}", r[4], r[5]]))

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write("\t".join(header) + "\n")
            for row in rows:
                fh.write("\t".join(row) + "\n")
        print(f"# salvo: {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
