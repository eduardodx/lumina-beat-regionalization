#!/usr/bin/env python3
"""Alinhamento-piloto + sonda de cobertura de UMA amostra (Ion + Nanopore) contra o hg38.

Responde as perguntas que só o alinhamento fecha (ver HANDOFF_SEQUENCIAMENTO_LAB_GENETICA.md):
  * escopo do Nanopore: targeted-BRCA vs. espalhado (WGS) -> onde os reads caem (cobertura por cromossomo)
  * viabilidade do variant calling -> profundidade/breadth em BRCA1 (chr17) e BRCA2 (chr13)
  * sanidade do pipeline: % de reads mapeados por plataforma

Encontra os FASTQ pela sample id nas subpastas `ion_torrent/` e `nanopore/` do --data-dir (evita copiar
os nomes enormes). Alinha com minimap2 (`-ax sr` p/ Ion curto, `-ax map-ont` p/ Nanopore), sorta/indexa,
e mede cobertura com `samtools coverage`. NÃO chama variantes — é só diagnóstico. Escala depois p/ as 6
amostras pareadas trocando --sample.

Requer no PATH (env seqlab): minimap2, samtools. hg38 em --ref (gera .fai se faltar).

    conda run -n seqlab python seq_pipeline/pilot_align_probe.py \
        --data-dir ~/seqlab/data --sample 353 --ref ~/hg38/hg38.fa --out-dir ~/seqlab/aln --threads 8
"""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

# Janelas generosas em torno dos genes em GRCh38 (prefixo "chr" ajustado ao naming do fasta).
# TP53 incluído porque o piloto revelou que o Nanopore é amplicon de TP53 (não BRCA) — sondar os três
# em toda amostra confirma o padrão Ion=BRCA / Nanopore=TP53.
GENE_REGIONS = {
    "BRCA1": ("17", 43_000_000, 43_180_000),
    "BRCA2": ("13", 32_310_000, 32_410_000),
    "TP53": ("17", 7_660_000, 7_695_000),
}


def run(cmd: list[str], timeout: int = 3600) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def need(tool: str) -> None:
    if shutil.which(tool) is None:
        sys.exit(f"ERRO: {tool} não está no PATH — ative o env seqlab (conda install ... minimap2 samtools).")


def find_fastq(data_dir: Path, subdir: str, sample: str) -> Path | None:
    hits = sorted((data_dir / subdir).glob(f"**/{sample}[_.]*"))
    hits = [h for h in hits if h.is_file() and h.suffix.lower() in (".fastq", ".fq")]
    return hits[0] if hits else None


def chr_prefix(fai: Path) -> str:
    first = fai.read_text(encoding="utf-8").splitlines()[0].split("\t")[0]
    return "chr" if first.startswith("chr") else ""


def align(fastq: Path, ref: Path, preset: str, out_bam: Path, threads: int) -> dict[str, Any]:
    log = out_bam.with_suffix(".minimap2.log")
    pipe = (
        f"minimap2 -ax {preset} -t {threads} {shlex.quote(str(ref))} {shlex.quote(str(fastq))} "
        f"2>{shlex.quote(str(log))} | samtools sort -@ {threads} -o {shlex.quote(str(out_bam))} - "
        f"&& samtools index {shlex.quote(str(out_bam))}"
    )
    r = run(["bash", "-c", pipe])
    if r.returncode != 0:
        return {"ok": False, "error": (r.stderr or log.read_text(errors="ignore")[-400:]).strip()[:400]}
    fs = run(["samtools", "flagstat", str(out_bam)])
    info: dict[str, Any] = {"ok": True, "bam": str(out_bam), "flagstat": fs.stdout.strip()}
    import re
    m = re.search(r"(\d+) \+ \d+ mapped \(([\d.]+|nan)%", fs.stdout)
    if m:
        info["mapped_reads"] = int(m.group(1))
        info["mapped_pct"] = m.group(2)
    return info


def coverage_by_chrom(bam: Path, top: int = 12) -> list[dict[str, Any]]:
    r = run(["samtools", "coverage", str(bam)])
    rows = []
    for line in r.stdout.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        f = line.split("\t")
        if len(f) < 7:
            continue
        try:
            rows.append({"chrom": f[0], "numreads": int(f[3]), "coverage_pct": float(f[5]), "meandepth": float(f[6])})
        except ValueError:
            continue
    rows.sort(key=lambda x: x["numreads"], reverse=True)
    return rows[:top]


def coverage_region(bam: Path, region: str) -> dict[str, Any]:
    r = run(["samtools", "coverage", "-r", region, str(bam)])
    for line in r.stdout.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        f = line.split("\t")
        if len(f) >= 7:
            try:
                return {"region": region, "numreads": int(f[3]), "coverage_pct": float(f[5]), "meandepth": float(f[6])}
            except ValueError:
                pass
    return {"region": region, "error": (r.stderr or "sem saída").strip()[:200]}


def probe_platform(name: str, fastq: Path, ref: Path, preset: str, out_bam: Path, threads: int,
                   pref: str) -> dict[str, Any]:
    print(f"[{name}] alinhando {fastq.name} (minimap2 -ax {preset})…", flush=True)
    res = align(fastq, ref, preset, out_bam, threads)
    if not res.get("ok"):
        print(f"[{name}] FALHOU: {res.get('error')}", flush=True)
        return {"platform": name, "fastq": str(fastq), **res}
    res["platform"] = name
    res["fastq"] = str(fastq)
    res["top_chroms"] = coverage_by_chrom(out_bam)
    res["genes"] = {g: coverage_region(out_bam, f"{pref}{c}:{s}-{e}") for g, (c, s, e) in GENE_REGIONS.items()}
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--sample", required=True, help="ex.: 353 (busca 353_* nas subpastas ion_torrent/ e nanopore/)")
    ap.add_argument("--ref", type=Path, required=True, help="hg38.fa")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args(argv)

    for t in ("minimap2", "samtools"):
        need(t)
    fai = args.ref.with_suffix(args.ref.suffix + ".fai")
    if not fai.exists():
        print(f"indexando {args.ref} (samtools faidx)…", flush=True)
        if run(["samtools", "faidx", str(args.ref)]).returncode != 0:
            sys.exit("ERRO: samtools faidx falhou.")
    pref = chr_prefix(fai)

    ion = find_fastq(args.data_dir, "ion_torrent", args.sample)
    ont = find_fastq(args.data_dir, "nanopore", args.sample)
    if ion is None and ont is None:
        sys.exit(f"Nenhum FASTQ para a amostra {args.sample} em {args.data_dir}/(ion_torrent|nanopore).")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {"sample": args.sample, "ref": str(args.ref), "chr_prefix": pref or "(numérico)"}
    if ion is not None:
        results["ion"] = probe_platform("Ion", ion, args.ref, "sr", args.out_dir / f"{args.sample}_ion.bam", args.threads, pref)
    else:
        print(f"[Ion] sem FASTQ para {args.sample}", flush=True)
    if ont is not None:
        results["nanopore"] = probe_platform("Nanopore", ont, args.ref, "map-ont", args.out_dir / f"{args.sample}_ont.bam", args.threads, pref)
    else:
        print(f"[Nanopore] sem FASTQ para {args.sample}", flush=True)

    out = args.out_dir / f"{args.sample}_pilot.json"
    out.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")

    # Resumo legível (é o que se cola de volta).
    print("=" * 72)
    print(f"PILOTO — amostra {args.sample} vs hg38 (prefixo contig: {results['chr_prefix']})")
    for key in ("ion", "nanopore"):
        r = results.get(key)
        if not r:
            continue
        if not r.get("ok"):
            print(f"[{r['platform']}] FALHOU: {r.get('error')}")
            continue
        print("-" * 72)
        print(f"[{r['platform']}] mapeados={r.get('mapped_reads','?')} ({r.get('mapped_pct','?')}%)")
        tops = ", ".join(f"{c['chrom']}:{c['numreads']}rd/{c['meandepth']:.0f}x" for c in r["top_chroms"][:6])
        print(f"  onde caem (top): {tops}")
        for g, cov in r["genes"].items():
            if "error" in cov:
                print(f"  {g} {cov['region']}: ERRO {cov['error']}")
            else:
                print(f"  {g} {cov['region']}: {cov['numreads']} reads | breadth {cov['coverage_pct']:.1f}% | meandepth {cov['meandepth']:.1f}x")
    print("-" * 72)
    print(f"JSON: {out}")
    print("Leitura esperada: Ion cobre BRCA1/BRCA2 (nao TP53); Nanopore cobre TP53 (nao BRCA). "
          "meandepth no gene-alvo = viabilidade do calling (Clair3 gosta de >=20-30x).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
