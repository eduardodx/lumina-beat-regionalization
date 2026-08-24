#!/usr/bin/env python3
"""Caracterização READ-ONLY dos dados de sequenciamento (frente lab de genética).

Resolve pelos DADOS as dúvidas técnicas do handoff (ver HANDOFF_SEQUENCIAMENTO_LAB_GENETICA.md §5):
  * #5 formato real (BAM/uBAM/FASTQ)      -> `file` + `samtools view -H`
  * #6 build (hg19/hg38/T2T)              -> @SQ do BAM: comprimento de chr1 é assinatura única
  * plataforma (Ion curto vs Nanopore longo) -> @RG PL do BAM e/ou comprimento médio dos reads
  * #4 mesmas amostras nas 2 plataformas  -> cruzamento de sample IDs (matriz amostra × plataforma)
  * #7 o que o `_final_clean` fez          -> stats bruto vs clean por amostra (nº reads, comprimento)
  * #2 painel vs genoma (parcial)          -> distribuição de reads por cromossomo (se BAM alinhado)

NÃO modifica, alinha, nem chama variantes — só lê metadados/estatísticas. Seguro para dado de paciente
(não escreve nada além do JSON/relatório de saída). Não sobe nada para lugar nenhum.

Requer no PATH: `samtools` e `seqkit` (env bioconda dedicado — ver o runbook). Roda com python3 stdlib.

    python seq_pipeline/characterize_seq_data.py --data-dir ~/seqlab/data --out ~/seqlab/characterization.json
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

# chr1 é assinatura única de build (comprimento em bp). Resolve #6 sem ambiguidade.
BUILD_BY_CHR1_LEN = {
    248956422: "GRCh38/hg38",
    249250621: "GRCh37/hg19",
    248387328: "T2T-CHM13v2.0",
}
# @RG PL -> plataforma legível.
PL_MAP = {
    "IONTORRENT": "Ion Torrent", "PGM": "Ion Torrent", "IONPROTON": "Ion Torrent",
    "ONT": "Nanopore", "OXFORD_NANOPORE": "Nanopore", "NANOPORE": "Nanopore",
    "ILLUMINA": "Illumina", "PACBIO": "PacBio", "BGI": "BGI", "MGI": "MGI",
}
FASTQ_RE = re.compile(r"\.(fastq|fq)(\.gz)?$", re.I)


def run(cmd: list[str], timeout: int = 900) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def have(tool: str) -> bool:
    return shutil.which(tool) is not None


def sample_id(name: str) -> str:
    """ID canônico da amostra a partir do nome do arquivo. 353_ALLS-R_... -> '353';
    353_final_clean.fastq -> '353'; Amostra21_sarcoma_... -> 'Amostra21'."""
    m = re.match(r"(\d+)", name)
    if m:
        return m.group(1)
    m = re.match(r"(amostra\s*\d+)", name, re.I)
    if m:
        return re.sub(r"\s+", "", m.group(1))
    return name.split("_")[0].split(".")[0]


def is_clean(name: str) -> bool:
    return "clean" in name.lower()


def platform_from_len(avg_len: float | None) -> str:
    if avg_len is None:
        return "desconhecida"
    if avg_len >= 600:
        return "long-read (provável Nanopore)"
    return "short/medium (provável Ion Torrent)"


def bam_info(path: Path) -> dict[str, Any]:
    info: dict[str, Any] = {"kind": "bam"}
    if not have("samtools"):
        info["error"] = "samtools ausente no PATH"
        return info
    info["quickcheck_ok"] = run(["samtools", "quickcheck", str(path)]).returncode == 0
    hdr = run(["samtools", "view", "-H", str(path)])
    if hdr.returncode != 0:
        info["error"] = (hdr.stderr or "header ilegível").strip()[:300]
        return info
    header = hdr.stdout

    sq: dict[str, int] = {}
    for line in header.splitlines():
        if line.startswith("@SQ"):
            fields = dict(f.split(":", 1) for f in line.split("\t")[1:] if ":" in f)
            if "SN" in fields and "LN" in fields:
                try:
                    sq[fields["SN"]] = int(fields["LN"])
                except ValueError:
                    pass
    info["n_contigs"] = len(sq)
    chr1 = sq.get("chr1") or sq.get("1")
    info["chr1_len"] = chr1
    if not sq:
        info["build_guess"] = "SEM @SQ — uBAM (não-alinhado)"
    else:
        info["build_guess"] = BUILD_BY_CHR1_LEN.get(chr1, f"desconhecido (chr1={chr1})") if chr1 else "sem chr1"
    info["contig_naming"] = "chr-prefixed" if any(k.startswith("chr") for k in sq) else ("numeric" if sq else "n/a")

    info["PG"] = [l[:500] for l in header.splitlines() if l.startswith("@PG")][:5]
    rg_lines = [l for l in header.splitlines() if l.startswith("@RG")]
    info["RG"] = [l[:400] for l in rg_lines][:5]
    pls = re.findall(r"\bPL:([^\t]+)", "\n".join(rg_lines))
    info["platform_tag"] = sorted({PL_MAP.get(p.upper(), p) for p in pls}) or None

    fs = run(["samtools", "flagstat", str(path)])
    if fs.returncode == 0:
        info["flagstat"] = fs.stdout.strip()
        m = re.search(r"(\d+) \+ \d+ mapped \(([\d.]+)%", fs.stdout)
        if m:
            info["mapped_pct"] = float(m.group(2))
    # Distribuição por cromossomo (painel? concentração) — amostra dos primeiros 200k reads (rápido, read-only).
    if info.get("quickcheck_ok"):
        view = run(["bash", "-c", f"samtools view {path} 2>/dev/null | head -200000 | cut -f3 | sort | uniq -c | sort -rn | head -25"])
        if view.returncode == 0 and view.stdout.strip():
            info["top_contigs_first200k"] = [ln.strip() for ln in view.stdout.strip().splitlines()]
    return info


def fastq_info(path: Path) -> dict[str, Any]:
    info: dict[str, Any] = {"kind": "fastq"}
    if not have("seqkit"):
        info["error"] = "seqkit ausente no PATH"
        return info
    st = run(["seqkit", "stats", "-a", "-T", str(path)])
    if st.returncode != 0:
        info["error"] = (st.stderr or "seqkit falhou").strip()[:300]
        return info
    lines = st.stdout.strip().splitlines()
    if len(lines) >= 2:
        stats = dict(zip(lines[0].split("\t"), lines[1].split("\t")))
        info["stats"] = stats
        try:
            info["avg_len"] = float(stats.get("avg_len", "nan"))
        except ValueError:
            info["avg_len"] = None
        info["platform_guess"] = platform_from_len(info.get("avg_len"))
    return info


def characterize_file(path: Path) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "path": str(path),
        "name": path.name,
        "size_mb": round(path.stat().st_size / 1e6, 1),
        "sample_id": sample_id(path.name),
        "is_clean": is_clean(path.name),
    }
    ftype = run(["file", "-b", str(path)]).stdout.strip() if have("file") else ""
    entry["file_magic"] = ftype[:120]
    if path.suffix.lower() == ".bam" or "BAM" in ftype:
        entry.update(bam_info(path))
    elif FASTQ_RE.search(path.name) or "FASTQ" in ftype.upper():
        entry.update(fastq_info(path))
    else:
        entry["kind"] = "outro/desconhecido"
    return entry


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, required=True, help="diretório com os dados (recursivo)")
    ap.add_argument("--out", type=Path, default=None, help="JSON de saída (default: <data-dir>/characterization.json)")
    ap.add_argument("--glob", default="**/*", help="padrão de arquivos (default: tudo, recursivo)")
    args = ap.parse_args(argv)

    for tool in ("samtools", "seqkit"):
        if not have(tool):
            print(f"AVISO: {tool} não está no PATH — ative o env bioconda (ver runbook).", file=sys.stderr)

    files = [p for p in sorted(args.data_dir.glob(args.glob))
             if p.is_file() and (p.suffix.lower() == ".bam" or FASTQ_RE.search(p.name))]
    if not files:
        print(f"Nenhum .bam/.fastq encontrado em {args.data_dir}", file=sys.stderr)
        return 1

    entries = [characterize_file(p) for p in files]

    # Agregação por amostra (resolve #4) e bruto-vs-clean (#7).
    by_sample: dict[str, dict[str, Any]] = defaultdict(lambda: {"raw": [], "clean": [], "platforms": set(), "builds": set()})
    for e in entries:
        sid = e["sample_id"]
        bucket = "clean" if e["is_clean"] else "raw"
        by_sample[sid][bucket].append(e["name"])
        plat = e.get("platform_tag") or ([e["platform_guess"]] if e.get("platform_guess") else [])
        for p in plat:
            by_sample[sid]["platforms"].add(p)
        if e.get("build_guess"):
            by_sample[sid]["builds"].add(e["build_guess"])
    sample_matrix = {sid: {**v, "platforms": sorted(v["platforms"]), "builds": sorted(v["builds"])}
                     for sid, v in by_sample.items()}

    builds_seen = sorted({e.get("build_guess") for e in entries if e.get("build_guess")})
    formats = sorted({e.get("kind", "?") for e in entries})
    report = {
        "data_dir": str(args.data_dir),
        "n_files": len(entries),
        "formats_seen": formats,
        "builds_seen": builds_seen,
        "n_samples": len(sample_matrix),
        "samples_with_two_platforms": sorted(s for s, v in sample_matrix.items() if len(v["platforms"]) >= 2),
        "sample_matrix": sample_matrix,
        "files": entries,
    }

    out = args.out or (args.data_dir / "characterization.json")
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    # Resumo legível (é o que se cola de volta no chat).
    print("=" * 72)
    print(f"CARACTERIZAÇÃO — {len(entries)} arquivos, {report['n_samples']} amostras em {args.data_dir}")
    print(f"Formatos: {formats}  |  Builds detectados (#6): {builds_seen or ['(nenhum BAM alinhado)']}")
    print(f"Amostras nas DUAS plataformas (#4): {report['samples_with_two_platforms'] or '(nenhuma / a confirmar)'}")
    print("-" * 72)
    for e in entries:
        plat = e.get("platform_tag") or e.get("platform_guess") or "?"
        extra = e.get("build_guess", "") if e["kind"] == "bam" else (f"avg_len={e.get('avg_len')}" if e["kind"] == "fastq" else "")
        n = e.get("stats", {}).get("num_seqs") or (e.get("mapped_pct") and f"{e['mapped_pct']}% mapped") or ""
        print(f"  {e['name'][:46]:46} {e['size_mb']:>7}MB {e['kind']:5} plat={str(plat)[:28]:28} {extra} {n}")
    print("-" * 72)
    print(f"JSON completo: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
