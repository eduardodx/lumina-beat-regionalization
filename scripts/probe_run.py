#!/usr/bin/env python3
"""Fase B da sonda de embedding: o run completo. Roda na GPU (~45 min numa A10G).

Extrai, para cada SITIO distinto do manifesto e cada um dos 8 layouts (5 janelas centradas +
3 casadas), as 4 bases na posicao focal: vetores focais pre e pos RMSNorm, metricas por bloco
(post448 / pre448 / h_up / h_pure) e o perfil espacial binado.

Chaveado por SITIO, nao por variante: as 4 bases cobrem qualquer ALT do ClinVar naquela posicao,
entao sitios multialelicos (e a sobreposicao entre os bracos `curated` e `paired_pb_gold`) nao
pagam forward duplicado. O vinculo de volta para variante/rotulo/painel/braco fica em
`probe_sites.parquet`.

Retomavel: cada grupo (layout, janela) e salvo assim que termina e pulado se ja existir no disco.

SAIDAS (em --out-dir)
---------------------
    probe_metrics.parquet              tabela principal da analise (1 linha por sitio x base x layout)
    probe_sites.parquet                sitio -> variant_id, arm, label, painel, phyloP, ...
    probe_focal_<layout>_<bp>.npz      vetores focais [n_sitios, 4, 448] pre e pos
    probe_profile_<layout>_<bp>.npz    perfis [n_sitios, 4, n_bins] + metadados dos bins
    probe_run.json                     metadados do run + checagens de consistencia

USO
---
    PYTHONPATH="$WORK" "$PY" scripts/probe_run.py \
        --manifest ~/probe/probe_manifest.parquet --fasta ~/hg38/hg38.fa \
        --checkpoint "$R03_CKPT" --out-dir ~/probe/run
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.embedding_probe.windows import (  # noqa: E402
    CONSUMER_WINDOWS,
    EXPLORATORY_WINDOWS,
    WindowError,
    build_window,
    focal_offset,
    matched_focal_index,
)

ALL_WINDOWS = tuple(sorted(EXPLORATORY_WINDOWS + CONSUMER_WINDOWS))
DEFAULT_CKPT = (
    "s3://croma-bioai-lumina-artifacts-us-east-2/experiments/LUM-20260719-001/"
    "runs/R03/checkpoints/final/best_checkpoint.pt"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--fasta", type=Path, required=True)
    p.add_argument("--checkpoint", default=DEFAULT_CKPT)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    p.add_argument("--allow-tf32", action="store_true")
    p.add_argument("--limit-sites", type=int, default=0, help="0 = todos; >0 trunca (para testar)")
    p.add_argument("--force", action="store_true", help="reprocessa grupos ja salvos")
    return p.parse_args(argv)


def layouts() -> list[tuple[int, int, str]]:
    """Os 8 layouts. `4096/matched` coincide com `4096/centered` (o indice focal casado E o offset
    centrado de 4096) -- mantido de proposito: custa ~2 min e serve de checagem de que os dois
    caminhos de codigo concordam."""
    matched = matched_focal_index(CONSUMER_WINDOWS)
    return ([(bp, focal_offset(bp), "centered") for bp in ALL_WINDOWS]
            + [(bp, matched, "matched") for bp in CONSUMER_WINDOWS])


def site_table(manifest):
    """Colapsa o manifesto em sitios distintos, agregando os vinculos por sitio."""
    import pandas as pd

    manifest = manifest.copy()
    manifest["site"] = (manifest["chrom"].astype(str) + ":"
                        + manifest["pos_1based"].astype(str) + ":" + manifest["ref"].astype(str))
    agg = manifest.groupby("site", sort=True).agg(
        chrom=("chrom", "first"),
        pos_1based=("pos_1based", "first"),
        ref=("ref", "first"),
        n_variants=("variant_id", "nunique"),
        clinvar_alts=("alt", lambda s: ",".join(sorted(set(s)))),
        variant_ids=("variant_id", lambda s: ",".join(sorted(set(s)))),
        arms=("arm", lambda s: ",".join(sorted(set(s)))),
        label_tiers=("label_tier", lambda s: ",".join(sorted(set(s)))),
        panels=("primary_panel", lambda s: ",".join(sorted(set(s)))),
        labels=("binary_label", lambda s: ",".join(str(x) for x in sorted(set(s)))),
        region_class=("region_class", "first"),
        phylop_241way=("phylop_241way", "first"),
        mane_gene=("mane_gene", "first"),
        gnomad_af_bin=("gnomad_af_bin", "first"),
        br_lab_any=("br_lab_any", "any"),
    ).reset_index()
    # Mapa alelo -> rotulo, para a analise pareada P-vs-B saber qual ALT e qual.
    lab = (manifest.groupby(["site", "alt"])["binary_label"].first().reset_index()
           .groupby("site").apply(lambda d: ";".join(f"{r.alt}={r.binary_label}" for r in d.itertuples()),
                                  include_groups=False))
    agg["alt_labels"] = agg["site"].map(lab)
    return agg


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    import numpy as np
    import pandas as pd
    import torch
    from pyfaidx import Fasta

    from eval.embedding_probe.extract import BASES, ProbeModel, extract_site, reducer_for

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fasta = Fasta(str(args.fasta), as_raw=True, sequence_always_upper=True)

    def fetch(chrom: str, start: int, end: int) -> str:
        if start < 0 or chrom not in fasta:
            return ""
        return str(fasta[chrom][start:end])

    manifest = pd.read_parquet(args.manifest)
    sites = site_table(manifest)
    if args.limit_sites:
        sites = sites.head(args.limit_sites)
    sites.to_parquet(args.out_dir / "probe_sites.parquet", index=False)
    print(f"[run] {len(manifest):,} linhas de manifesto -> {len(sites):,} sitios distintos")
    print(f"[run] bracos: {dict(manifest['arm'].value_counts())}")

    probe = ProbeModel(args.checkpoint, args.device,
                       dtype=getattr(torch, args.dtype), allow_tf32=args.allow_tf32)
    print(f"[run] R03 carregado | params={sum(p.numel() for p in probe.model.parameters()):,}")

    meta = {
        "checkpoint": args.checkpoint, "dtype": args.dtype, "allow_tf32": args.allow_tf32,
        "device": args.device, "n_sites": int(len(sites)), "n_manifest_rows": int(len(manifest)),
        "layouts": [f"{lay}_{bp}" for bp, _, lay in layouts()], "groups": {}, "checks": {},
    }
    all_metrics: list[dict] = []
    t_start = time.perf_counter()

    for window_bp, focal_index, layout in layouts():
        group = f"{layout}_{window_bp}"
        focal_path = args.out_dir / f"probe_focal_{group}.npz"
        prof_path = args.out_dir / f"probe_profile_{group}.npz"
        metrics_path = args.out_dir / f"probe_metrics_{group}.parquet"
        if focal_path.exists() and metrics_path.exists() and not args.force:
            print(f"[run] {group}: ja existe, pulando")
            all_metrics.append(pd.read_parquet(metrics_path))
            continue

        reducer = reducer_for(window_bp, focal_index, probe.device)
        n_bins = reducer.n_bins
        n = len(sites)
        focal_pre = np.zeros((n, 4, probe.d_full), np.float32)
        focal_post = np.zeros((n, 4, probe.d_full), np.float32)
        prof = {k: np.zeros((n, 4, n_bins), np.float32)
                for k in ("hup_mean", "hup_max", "hpure_mean", "hpure_max", "post_mean", "post_max")}
        rows: list[dict] = []
        failures: list[dict] = []
        t0 = time.perf_counter()

        for i, site in enumerate(sites.itertuples()):
            try:
                w = build_window(fetch, chrom=site.chrom, pos_1based=int(site.pos_1based),
                                 ref=site.ref, alt=BASES[0] if site.ref != BASES[0] else BASES[1],
                                 window_bp=window_bp,
                                 focal_index=None if layout == "centered" else focal_index)
            except WindowError as exc:
                failures.append({"site": site.site, "reason": str(exc)})
                continue
            res = extract_site(probe, ref_seq=w.ref_seq, focal_index=w.focal_index, reducer=reducer)
            focal_pre[i], focal_post[i] = res["pre_focal"], res["post_focal"]
            for base, p in res["profiles"].items():
                for key, arr in p.items():
                    prof[key][i, BASES.index(base)] = arr
            for base, m in res["metrics"].items():
                rows.append({"site": site.site, "chrom": site.chrom, "pos_1based": int(site.pos_1based),
                             "ref": res["ref_base"], "base": base, "window_bp": window_bp,
                             "layout": layout, "focal_index": w.focal_index,
                             "is_clinvar_alt": base in str(site.clinvar_alts).split(","), **m})
            if (i + 1) % 200 == 0:
                rate = (time.perf_counter() - t0) / (i + 1)
                print(f"    {group}: {i + 1}/{n}  ({rate:.3f} s/sitio, "
                      f"faltam ~{rate * (n - i - 1) / 60:.1f} min)", flush=True)

        elapsed = time.perf_counter() - t0
        bins = reducer.bins
        np.savez_compressed(focal_path, site=sites["site"].to_numpy().astype("U40"),
                            bases=np.array(BASES, dtype="U1"), pre=focal_pre, post=focal_post)
        np.savez_compressed(prof_path, site=sites["site"].to_numpy().astype("U40"),
                            bases=np.array(BASES, dtype="U1"),
                            bin_label=np.array([b.label for b in bins], dtype="U24"),
                            bin_direction=np.array([b.direction for b in bins], dtype="U8"),
                            bin_lo=np.array([b.lo for b in bins]), bin_hi=np.array([b.hi for b in bins]),
                            bin_width=np.array([b.end - b.start for b in bins]), **prof)
        df = pd.DataFrame(rows)
        df.to_parquet(metrics_path, index=False)
        all_metrics.append(df)
        meta["groups"][group] = {"seconds": elapsed, "n_sites": n, "n_rows": len(df),
                                 "n_bins": n_bins, "n_failures": len(failures),
                                 "failures": failures[:20]}
        print(f"[run] {group}: {elapsed / 60:.1f} min | {len(df):,} linhas | "
              f"{len(failures)} falhas | -> {focal_path.name}")

    metrics = pd.concat(all_metrics, ignore_index=True)
    metrics.to_parquet(args.out_dir / "probe_metrics.parquet", index=False)

    # --- checagens de consistencia ------------------------------------------------------------
    print("\n[run] checagens de consistencia")
    a = np.load(args.out_dir / "probe_focal_centered_4096.npz")
    b = np.load(args.out_dir / "probe_focal_matched_4096.npz")
    same = float(np.abs(a["pre"] - b["pre"]).max())
    print(f"  4096 centered vs matched (a MESMA janela por construcao): max|diff|={same:.3e} -> "
          f"{'OK' if same == 0.0 else 'DIVERGEM -- os dois caminhos de codigo nao concordam'}")
    meta["checks"]["centered_vs_matched_4096"] = {"max_abs": same, "ok": same == 0.0}

    hpure_off = 0.0
    for window_bp, _, layout in layouts():
        p = np.load(args.out_dir / f"probe_profile_{layout}_{window_bp}.npz")
        non_focal = p["bin_direction"] != "focal"
        hpure_off = max(hpure_off, float(np.abs(p["hpure_max"][:, :, non_focal]).max()))
    print(f"  Delta h_pure fora do focal, todos os layouts: max={hpure_off:.3e} -> "
          f"{'OK (zero)' if hpure_off == 0.0 else 'FALHOU'}")
    meta["checks"]["hpure_off_focal"] = {"max_abs": hpure_off, "ok": hpure_off == 0.0}

    meta["total_minutes"] = (time.perf_counter() - t_start) / 60
    probe.close()
    (args.out_dir / "probe_run.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(f"\n[run] {meta['total_minutes']:.1f} min no total | {len(metrics):,} linhas de metrica")
    print(f"[run] saidas em {args.out_dir}")
    print("[run] traz de volta: probe_metrics.parquet, probe_sites.parquet, probe_run.json "
          "e os probe_profile_*.npz (os focal_*.npz sao grandes; so se formos analisar vetores).")
    return 0 if meta["checks"]["centered_vs_matched_4096"]["ok"] and meta["checks"]["hpure_off_focal"]["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
