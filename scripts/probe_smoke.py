#!/usr/bin/env python3
"""Fase 2 da sonda de embedding: portao de validacao + medicao de custo. Roda na GPU.

Nao produz resultado cientifico. Ele existe para PROVAR as premissas do desenho antes de gastar
o run completo, e para medir o custo por janela e dimensionar o resto. Sete checagens:

  1. **Hook**: ``RMSNorm(pre)`` reproduz o ``last_hidden_state`` retornado. Sem isso, "capturamos
     o tensor pre-norm" seria suposicao -- e toda a decomposicao h_up/h_pure depende dela.
  2. **Independencia de batch**: a linha `ref` sozinha e dentro de um batch de 4 tem que ser
     identica. Colocamos as 4 bases num batch so; se houvesse qualquer interacao entre linhas,
     os deltas estariam contaminados.
  3. **Determinismo intra-batch**: 4 copias da MESMA sequencia -> 4 linhas identicas.
  4. **Determinismo entre chamadas**: mesma entrada, duas chamadas -> mesma saida. Kernels do
     Mamba nao sao obrigados a ser deterministicos; se nao forem, o piso de ruido entra na
     interpretacao de todo Delta.
  5. **Invariante analitico de h_pure**: Delta h_pure e zero fora do focal e constante por par
     (ref,alt) em qualquer locus/janela. Previsao exata da arquitetura; falhar aqui = bug nosso.
  6. **O canal trivial existe e e grande**: compara Delta em h_pure (trivial) vs h_up (contextual).
     Quantifica quanto de "o embedding mudou" e explicado por construcao.
  7. **Custo**: tempo e pico de memoria por (janela, layout), para dimensionar a Fase B.

USO
---
    export R03_CKPT="s3://croma-bioai-lumina-artifacts-us-east-2/experiments/LUM-20260719-001/runs/R03/checkpoints/final/best_checkpoint.pt"
    PYTHONPATH="$WORK" "$PY" scripts/probe_smoke.py \
        --manifest ~/probe/probe_manifest.parquet \
        --fasta ~/hg38/hg38.fa \
        --checkpoint "$R03_CKPT" \
        --out ~/probe/probe_smoke.json
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
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n-sites", type=int, default=8, help="sitios do smoke (>=2 por base de REF)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    p.add_argument("--allow-tf32", action="store_true",
                   help="liga TF32 nas convolucoes/matmuls (mais rapido, ~10 bits de mantissa)")
    return p.parse_args(argv)


def layouts() -> list[tuple[int, int, str]]:
    matched = matched_focal_index(CONSUMER_WINDOWS)
    return ([(bp, focal_offset(bp), "centered") for bp in ALL_WINDOWS]
            + [(bp, matched, "matched") for bp in CONSUMER_WINDOWS])


def pick_sites(manifest, n: int) -> list[dict]:
    """Sitios distintos, cobrindo as 4 bases de REF com >=2 cada quando possivel.

    Com >=2 sitios por base de REF, cada tipo de substituicao aparece em loci diferentes -- que e
    exatamente o que o invariante de h_pure (constancia por par ref,alt) precisa para ser testavel.
    """
    seen: set[tuple] = set()
    by_ref: dict[str, list[dict]] = {}
    for row in manifest.to_dict("records"):
        key = (row["chrom"], int(row["pos_1based"]), row["ref"])
        if key in seen:
            continue
        seen.add(key)
        by_ref.setdefault(row["ref"], []).append(row)
    picked: list[dict] = []
    used: set[tuple] = set()

    def key_of(row: dict) -> tuple:
        return (row["chrom"], int(row["pos_1based"]), row["ref"])

    for base in "ACGT":  # passe 1: ate 2 por base de REF
        for row in by_ref.get(base, [])[:2]:
            if len(picked) < n:
                picked.append(row)
                used.add(key_of(row))
    for base in "ACGT":  # passe 2: completa com o que sobrou, sem repetir sitio
        for row in by_ref.get(base, []):
            if len(picked) >= n:
                return picked
            if key_of(row) not in used:
                picked.append(row)
                used.add(key_of(row))
    return picked


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    import numpy as np
    import pandas as pd
    import torch
    from pyfaidx import Fasta

    from eval.embedding_probe.extract import (
        BASES, ProbeModel, check_hpure_invariant, extract_site, reducer_for,
    )

    fasta = Fasta(str(args.fasta), as_raw=True, sequence_always_upper=True)

    def fetch(chrom: str, start: int, end: int) -> str:
        if start < 0 or chrom not in fasta:
            return ""
        return str(fasta[chrom][start:end])

    manifest = pd.read_parquet(args.manifest)
    sites = pick_sites(manifest, args.n_sites)
    print(f"[smoke] {len(sites)} sitios: "
          + ", ".join(f"{s['chrom']}:{s['pos_1based']}{s['ref']}" for s in sites))

    dtype = getattr(torch, args.dtype)
    print(f"[smoke] carregando R03 ({args.dtype}, tf32={args.allow_tf32}) em {args.device} ...")
    t0 = time.perf_counter()
    probe = ProbeModel(args.checkpoint, args.device, dtype=dtype, allow_tf32=args.allow_tf32)
    n_params = sum(p.numel() for p in probe.model.parameters())
    print(f"[smoke] carregado em {time.perf_counter() - t0:.1f}s | params={n_params:,} "
          f"| d_model={probe.d_model} d_pure={probe.d_pure} d_full={probe.d_full} l_max={probe.l_max:,}")
    if n_params != 52_124_400:
        print(f"  [!!] esperado 52.124.400 params no R03; veio {n_params:,}")

    report: dict = {
        "checkpoint": args.checkpoint, "device": args.device, "dtype": args.dtype,
        "allow_tf32": args.allow_tf32, "n_params": n_params,
        "d_model": probe.d_model, "d_pure": probe.d_pure, "l_max": probe.l_max,
        "sites": [f"{s['chrom']}:{s['pos_1based']}:{s['ref']}" for s in sites],
        "checks": {}, "cost": {},
    }
    cuda = args.device.startswith("cuda")
    site0 = sites[0]

    # --- 1..4: checagens estruturais, na janela pequena (baratas) --------------------------
    win = build_window(fetch, chrom=site0["chrom"], pos_1based=int(site0["pos_1based"]),
                       ref=site0["ref"], alt=site0["alt"], window_bp=4096)
    ref_seq, focal = win.ref_seq, win.focal_index
    four = [ref_seq[:focal] + b + ref_seq[focal + 1:] for b in BASES]

    print("\n[1] hook pre-RMSNorm")
    hook = probe.verify_hook(four)
    print(f"    RMSNorm(pre) vs last_hidden_state: max|diff|={hook['max_abs_diff']:.3e} -> "
          f"{'OK' if hook['ok'] else 'FALHOU'}")
    report["checks"]["hook"] = hook

    print("[2] invariancia de CONTEUDO a batch fixo (o que o pipeline exige)")
    def as_np(pair):
        return [t.detach().float().cpu().numpy().copy() for t in pair]

    # A linha do `ref` tem que ser identica independentemente do CONTEUDO das outras linhas do
    # batch. E isso que autoriza extrair as 4 bases juntas. O efeito de TAMANHO de batch (B=1 vs
    # B=4) e outra coisa -- numerico, medido abaixo e em probe_batch_diagnostic.py -- e NAO e
    # bloqueante, porque a sonda usa batch fixo de 4 e compara ref/alt dentro do mesmo forward.
    pre_mixed, post_mixed = as_np(probe.encode([ref_seq] + four[1:]))
    pre_same, post_same = as_np(probe.encode([ref_seq] * 4))
    d_pre = float(np.abs(pre_mixed[0] - pre_same[0]).max())
    d_post = float(np.abs(post_mixed[0] - post_same[0]).max())
    print(f"    linha do ref, conteudo vizinho diferente: pre={d_pre:.3e} post={d_post:.3e} -> "
          f"{'OK (sem cross-talk)' if max(d_pre, d_post) == 0.0 else 'HA cross-talk entre linhas'}")
    report["checks"]["batch_content_invariance"] = {
        "pre": d_pre, "post": d_post, "ok": max(d_pre, d_post) == 0.0
    }

    pre_alone, _ = as_np(probe.encode([ref_seq]))
    row = BASES.index(ref_seq[focal])
    size_effect = float(np.linalg.norm(pre_alone[0, focal] - pre_mixed[row, focal]))
    print(f"    [informativo] efeito de TAMANHO de batch (B=1 vs B=4) no focal: "
          f"||diff||={size_effect:.3e}")
    report["checks"]["batch_size_numerical_effect"] = {"focal_l2_b1_vs_b4": size_effect}

    print("[3] determinismo intra-batch (4 copias identicas)")
    pre_rep = as_np(probe.encode([ref_seq] * 4))[0]
    intra = float(np.abs(pre_rep - pre_rep[0:1]).max())
    print(f"    max|diff| entre linhas={intra:.3e} -> {'OK' if intra == 0.0 else 'nao-zero'}")
    report["checks"]["intra_batch_determinism"] = {"max_abs": intra, "ok": intra < 1e-6}

    print("[4] determinismo entre chamadas (o 'controle nulo' ref-vs-ref)")
    a_pre, a_post = as_np(probe.encode(four))
    b_pre, b_post = as_np(probe.encode(four))
    run_pre = float(np.abs(a_pre - b_pre).max())
    run_post = float(np.abs(a_post - b_post).max())
    print(f"    max|diff| pre={run_pre:.3e} post={run_post:.3e} -> "
          f"{'deterministico' if max(run_pre, run_post) == 0.0 else 'NAO deterministico (piso de ruido)'}")
    report["checks"]["run_to_run_determinism"] = {
        "pre": run_pre, "post": run_post, "deterministic": max(run_pre, run_post) == 0.0
    }

    # --- 5..7: extracao real em todos os layouts, com custo -------------------------------
    print("\n[5-7] extracao em todos os layouts")
    results: list[dict] = []
    for window_bp, focal_index, layout in layouts():
        reducer = reducer_for(window_bp, focal_index, probe.device)
        if cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        per_layout = []
        for site in sites:
            w = build_window(fetch, chrom=site["chrom"], pos_1based=int(site["pos_1based"]),
                             ref=site["ref"], alt=site["alt"], window_bp=window_bp,
                             focal_index=None if layout == "centered" else focal_index)
            res = extract_site(probe, ref_seq=w.ref_seq, focal_index=w.focal_index, reducer=reducer)
            res["window_bp"], res["layout"] = window_bp, layout
            per_layout.append(res)
        if cuda:
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        peak_gb = torch.cuda.max_memory_allocated() / 2**30 if cuda else float("nan")
        results.extend(per_layout)

        per_site = elapsed / len(sites)
        report["cost"][f"{layout}_{window_bp}"] = {
            "seconds_total": elapsed, "seconds_per_site": per_site, "peak_gb": peak_gb,
            "n_sites": len(sites), "forwards_per_site": 4,
        }
        print(f"    {layout:<9} {window_bp:>6} bp | {per_site:6.3f} s/sitio | pico {peak_gb:5.2f} GB")

    print("\n[5] invariante analitico de h_pure")
    inv = check_hpure_invariant(results)
    print(f"    Delta h_pure fora do focal: max={inv['off_focal_max_abs']:.3e} -> "
          f"{'OK (zero)' if inv['off_focal_ok'] else 'FALHOU -- extracao errada'}")
    print(f"    constancia por par (ref,alt) em {inv['n_substitutions']} substituicoes: "
          f"max spread={inv['substitution_spread_max']:.3e} -> "
          f"{'OK' if inv['substitution_ok'] else 'FALHOU -- extracao errada'}")
    report["checks"]["hpure_invariant"] = inv

    print("\n[6] tamanho do canal trivial vs contextual (janela 4096 centrada)")
    rows = [r for r in results if r["window_bp"] == 4096 and r["layout"] == "centered"]
    trivial, contextual = [], []
    for res in rows:
        for base, m in res["metrics"].items():
            trivial.append(m["hpure_cosine_distance"])
            contextual.append(m["hup_cosine_distance"])
    summary = {
        "hpure_cosine_median": float(np.median(trivial)),
        "hup_cosine_median": float(np.median(contextual)),
        "hpure_cosine_min": float(np.min(trivial)),
        "hup_cosine_min": float(np.min(contextual)),
        "n_pairs": len(trivial),
    }
    print(f"    cosine dist mediana -- h_pure (trivial) = {summary['hpure_cosine_median']:.4f}"
          f" | h_up (contextual) = {summary['hup_cosine_median']:.4f}")
    print(f"    minimos             -- h_pure = {summary['hpure_cosine_min']:.4f}"
          f" | h_up = {summary['hup_cosine_min']:.4f}")
    report["checks"]["trivial_vs_contextual"] = summary

    print()
    print("[7] previa do perfil espacial (4096 centrada, h_up)")
    # MEDIA e a estatistica de decaimento; o MAXIMO cresce com a largura do bin (max sobre 1024
    # posicoes e naturalmente maior que sobre 1 posicao), entao olhar so o max leria um plato
    # falso onde ha apenas mais amostras.
    res = rows[0]
    labels, directions = res["bin_labels"], res["bin_directions"]
    base0 = next(iter(res["profiles"]))
    prof_max = res["profiles"][base0]["hup_max"]
    prof_mean = res["profiles"][base0]["hup_mean"]
    wanted = ("focal", "down_1_1", "down_5_5", "down_10_10", "down_25_25",
              "down_26_32", "down_65_128", "down_257_512", "down_513_1024", "down_1025_2048")
    widths = {b.label: b.end - b.start
              for b in reducer_for(4096, focal_offset(4096), probe.device).bins}
    preview: dict = {}
    print(f"    {'bin':<16} {'n_bases':>8} {'media':>14} {'maximo':>14}")
    for label, direction, mean_v, max_v in zip(labels, directions, prof_mean, prof_max):
        if direction in ("focal", "down") and label in wanted:
            preview[label] = {"mean": float(mean_v), "max": float(max_v), "n_bases": widths[label]}
            print(f"    {label:<16} {widths[label]:>8} {mean_v:14.6e} {max_v:14.6e}")
    report["checks"]["profile_preview_hup"] = preview
    report["checks"]["profile_preview_base"] = f"{res['ref_base']}>{base0}"

    total = sum(v["seconds_per_site"] for v in report["cost"].values())
    n_sites_full = int(manifest[["chrom", "pos_1based", "ref"]].drop_duplicates().shape[0])
    report["projection"] = {
        "seconds_per_site_all_layouts": total,
        "n_sites_in_manifest": n_sites_full,
        "projected_hours": total * n_sites_full / 3600,
    }
    print(f"\n[custo] {total:.2f} s/sitio somando os 8 layouts")
    print(f"[custo] manifesto tem {n_sites_full:,} sitios -> "
          f"projecao ~{report['projection']['projected_hours']:.2f} h de GPU")

    probe.close()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(f"\n[smoke] relatorio: {args.out}")

    hard = ["hook", "batch_content_invariance", "intra_batch_determinism"]
    failed = [k for k in hard if not report["checks"][k].get("ok", False)]
    if not inv["off_focal_ok"] or not inv["substitution_ok"]:
        failed.append("hpure_invariant")
    if failed:
        print(f"\n[smoke] PORTAO FECHADO -- falharam: {failed}")
        return 1
    print("\n[smoke] portao aberto. Manda o probe_smoke.json.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
