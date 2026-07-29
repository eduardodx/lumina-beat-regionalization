#!/usr/bin/env python3
"""Fase 0.5 -- NIVEL 2: poder REAL do endpoint (ΔAUROC/ΔMCC/ΔAP) com os pares reais + gene-clustered.

RODA LOCAL (CPU). A a1 (nivel 1) deu o TETO: n suposto por cobertura + bootstrap variant-level, que
subestima a incerteza porque o br_only agrupa por gene. Este script fecha isso: le os pares REAIS do
matcher (`t_nonbr_matched.parquet`), usa o n real (3651) e o vetor de labels/genes reais, e reamostra
por GENE (cluster=gene_symbol, como o v11/Pedro). Da o poder honesto -- o real, nao o teto.

O que e REAL vs SIMULADO: reais = n_pares, o vetor de label por par, e o gene por par (pro cluster).
Simulados = os scores M1/M2 (os modelos ainda nao existem; a campanha M1-M4 e que os gera). Entao
isto mede DETECTABILIDADE dado o n/estrutura reais, sob um efeito de discriminacao (mesmo caveat da
a1: efeito puro de calibracao/cenario H so o MCC/Brier pegam).

USO
---
    python scripts/estimate_br_power_level2.py \\
        --pairs ~/artifacts/fase0/t_nonbr_matched.parquet \\
        --auc 0.90 --rho 0.8 --out ~/artifacts/fase05/a1_power_level2.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("a1_level2")

_NORM = NormalDist(0.0, 1.0)
METRICS = ("mcc", "auroc", "ap")
DEFAULT_EFFECT_GRID = (0.0, 0.10, 0.20, 0.30, 0.50)
METRIC_TARGET = {"mcc": 0.05, "auroc": 0.02, "ap": 0.02}


# --- metricas (identicas a a1) --------------------------------------------------------------------
def _mcc_rows(pred, y):
    tp = (pred * y).sum(axis=1); fp = (pred * (1 - y)).sum(axis=1)
    fn = ((1 - pred) * y).sum(axis=1); tn = ((1 - pred) * (1 - y)).sum(axis=1)
    num = tp * tn - fp * fn
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return np.where(den > 0, num / den, 0.0)


def _auroc_rows(scores, y):
    ranks = np.argsort(np.argsort(scores, axis=1), axis=1).astype(np.float64) + 1.0
    yf = y.astype(np.float64); n_pos = yf.sum(axis=1); n_neg = scores.shape[1] - n_pos
    rank_sum_pos = (ranks * yf).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        auc = (rank_sum_pos - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg)
    return np.where((n_pos > 0) & (n_neg > 0), auc, np.nan)


def _ap_rows(scores, y):
    order = np.argsort(-scores, axis=1)
    sorted_y = np.take_along_axis(y.astype(np.float64), order, axis=1)
    cum_pos = np.cumsum(sorted_y, axis=1)
    positions = np.arange(1, scores.shape[1] + 1, dtype=np.float64)[None, :]
    with np.errstate(invalid="ignore", divide="ignore"):
        ap = ((cum_pos / positions) * sorted_y).sum(axis=1) / sorted_y.sum(axis=1)
    return np.where(sorted_y.sum(axis=1) > 0, ap, np.nan)


def bc_interval(draws, observed, alpha=0.05):
    draws = draws[~np.isnan(draws)]
    if draws.size < 2:
        return (float("nan"), float("nan"))
    prop = min(max(float(np.mean(draws < observed)), 1e-6), 1 - 1e-6)
    z0 = _NORM.inv_cdf(prop)
    lo_p = _NORM.cdf(2 * z0 + _NORM.inv_cdf(alpha / 2))
    hi_p = _NORM.cdf(2 * z0 + _NORM.inv_cdf(1 - alpha / 2))
    return (float(np.percentile(draws, 100 * lo_p)), float(np.percentile(draws, 100 * hi_p)))


def _sim_pair_scores(y, auc, rho, rng, effect):
    n = len(y)
    mu = math.sqrt(2.0) * _NORM.inv_cdf(auc)
    shared = rng.standard_normal(n)
    a, b = math.sqrt(rho), math.sqrt(1.0 - rho)
    base = mu * y
    m1 = base + a * shared + b * rng.standard_normal(n)
    m2 = base + a * shared + b * rng.standard_normal(n) + effect * y
    return m1, m2


def _best_threshold(scores, y, n_grid=101):
    qs = np.quantile(scores, np.linspace(0.02, 0.98, n_grid))
    preds = (scores[None, :] >= qs[:, None]).astype(np.int64)
    return float(qs[int(np.argmax(_mcc_rows(preds, np.broadcast_to(y, preds.shape))))])


def _did_rows(y2d, m1br, m2br, m1nb, m2nb, thr1, thr2):
    out = {}

    def mcc(s, thr):
        return _mcc_rows((s >= thr).astype(np.int64), y2d)

    out["mcc"] = (mcc(m2br, thr2) - mcc(m1br, thr1)) - (mcc(m2nb, thr2) - mcc(m1nb, thr1))
    for name, fn in (("auroc", _auroc_rows), ("ap", _ap_rows)):
        out[name] = (fn(m2br, y2d) - fn(m1br, y2d)) - (fn(m2nb, y2d) - fn(m1nb, y2d))
    return out


# --- bootstrap GENE-CLUSTERED (a diferenca vs a1) -------------------------------------------------
def _gene_clustered_draws(y, m1br, m2br, m1nb, m2nb, thr1, thr2, gene_positions, gene_arr, boot, rng):
    draws = {m: np.empty(boot) for m in METRICS}
    n_genes = len(gene_arr)
    for b in range(boot):
        sampled = gene_arr[rng.integers(0, n_genes, size=n_genes)]
        idx = np.concatenate([gene_positions[g] for g in sampled])
        d = _did_rows(y[idx][None], m1br[idx][None], m2br[idx][None], m1nb[idx][None], m2nb[idx][None], thr1, thr2)
        for m in METRICS:
            draws[m][b] = d[m][0]
    return draws


def run_power_cell(y, genes, gene_positions, gene_arr, auc, rho, effect, campaigns, boot, rng):
    realized = {m: [] for m in METRICS}
    detected = {m: 0 for m in METRICS}
    for _ in range(campaigns):
        m1br, m2br = _sim_pair_scores(y, auc, rho, rng, effect)
        m1nb, m2nb = _sim_pair_scores(y, auc, rho, rng, 0.0)
        c1, _ = _sim_pair_scores(y, auc, rho, rng, 0.0)
        _, c2 = _sim_pair_scores(y, auc, rho, rng, 0.0)
        thr1, thr2 = _best_threshold(c1, y), _best_threshold(c2, y)
        pt = _did_rows(y[None], m1br[None], m2br[None], m1nb[None], m2nb[None], thr1, thr2)
        draws = _gene_clustered_draws(y, m1br, m2br, m1nb, m2nb, thr1, thr2, gene_positions, gene_arr, boot, rng)
        for m in METRICS:
            obs = float(pt[m][0])
            realized[m].append(obs)
            lo, _hi = bc_interval(draws[m], obs)
            if not math.isnan(lo) and lo > 0.0:
                detected[m] += 1
    return {m: (float(np.nanmean(realized[m])), detected[m] / campaigns) for m in METRICS}


def _interp(xs_ys, x_target):
    pts = sorted(xs_ys, key=lambda t: t[0])
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    if not xs:
        return float("nan")
    if x_target <= xs[0]:
        return ys[0]
    if x_target >= xs[-1]:
        return ys[-1]
    return float(np.interp(x_target, xs, ys))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pairs", required=True, type=Path, help="parquet dos pares do matcher")
    p.add_argument("--label-col", default="br_label")
    p.add_argument("--gene-col", default="br_GeneSymbol")
    p.add_argument("--auc", type=float, nargs="+", default=[0.90])
    p.add_argument("--rho", type=float, nargs="+", default=[0.8])
    p.add_argument("--mcc-anchor", type=float, default=0.05)
    p.add_argument("--campaigns", type=int, default=100)
    p.add_argument("--bootstrap", type=int, default=250)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    rng = np.random.default_rng(args.seed)

    df = pd.read_parquet(args.pairs)
    y = pd.to_numeric(df[args.label_col], errors="coerce").fillna(0).astype(np.int64).to_numpy()
    genes = df[args.gene_col].astype(str).to_numpy()
    n = len(y)
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    gene_arr = np.array(sorted(set(genes)))
    gene_positions = {g: np.where(genes == g)[0] for g in gene_arr}
    log.info("pares=%d (pos=%d neg=%d) | genes=%d | maior gene=%d pares",
             n, n_pos, n_neg, len(gene_arr), max(len(v) for v in gene_positions.values()))

    results = []
    for auc in args.auc:
        for rho in args.rho:
            curve = []
            for eff in DEFAULT_EFFECT_GRID:
                cell = run_power_cell(y, genes, gene_positions, gene_arr, auc, rho, eff, args.campaigns, args.bootstrap, rng)
                curve.append({"effect_latent": eff, **{m: {"realized": cell[m][0], "power": cell[m][1]} for m in METRICS}})
            mcc_x = [pt["mcc"]["realized"] for pt in curve]
            head2head = {m: _interp(list(zip(mcc_x, [pt[m]["power"] for pt in curve])), args.mcc_anchor) for m in METRICS}
            own = {m: _interp([(pt[m]["realized"], pt[m]["power"]) for pt in curve], METRIC_TARGET[m]) for m in METRICS}
            results.append({"auc": auc, "rho": rho, "curve": curve,
                            "head_to_head_at_mcc_anchor": head2head, "power_at_own_target": own})
            log.info("AUC=%.2f rho=%.2f | HEAD-TO-HEAD (gene-clustered, REAL) @ ΔMCC=%.2f: "
                     "MCC=%.2f AUROC=%.2f AP=%.2f", auc, rho, args.mcc_anchor,
                     head2head["mcc"], head2head["auroc"], head2head["ap"])
            log.info("   poder@alvo-proprio: MCC@0.05=%.2f  AUROC@0.02=%.2f  AP@0.02=%.2f",
                     own["mcc"], own["auroc"], own["ap"])

    payload = {"pairs": str(args.pairs), "n_pairs": n, "n_pos": n_pos, "n_neg": n_neg,
               "n_genes": int(len(gene_arr)), "bootstrap": "gene-clustered (cluster=GeneSymbol)",
               "mcc_anchor": args.mcc_anchor, "metric_targets": METRIC_TARGET, "results": results,
               "params": {"campaigns": args.campaigns, "bootstrap": args.bootstrap, "auc": args.auc, "rho": args.rho},
               "note": "Poder REAL (gene-clustered) -- compara com o TETO variant-level da a1. Efeito "
                       "modelado e de discriminacao (cenario H fica com MCC/Brier)."}
    print(json.dumps(payload, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2))
        log.info("Escrito %s", args.out)

    ref = results[0]
    h = ref["head_to_head_at_mcc_anchor"]
    log.info("=== VEREDITO NIVEL 2 (REAL, gene-clustered; AUC=%.2f) ===", ref["auc"])
    log.info("No mesmo efeito verdadeiro (ΔMCC=%.2f): poder MCC=%.2f vs AUROC=%.2f vs AP=%.2f",
             args.mcc_anchor, h["mcc"], h["auroc"], h["ap"])
    if h["auroc"] >= 0.7 and h["auroc"] - h["mcc"] >= 0.2:
        log.info("  => AUROC segue claramente melhor mesmo com o haircut gene-clustered. Troca confirmada no real.")
    elif h["auroc"] - h["mcc"] >= 0.1:
        log.info("  => AUROC ainda melhor, mas o haircut apertou; reportar o numero real (nao o teto) ao Eduardo.")
    else:
        log.info("  => O haircut gene-clustered nivelou as metricas; endpoint clinico fraco no real -> peso no representacional/AMR.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
