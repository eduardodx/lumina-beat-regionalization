#!/usr/bin/env python3
"""Fase 0.5 -- sonda (a1): o endpoint clinico tem PODER dado o n real do br_only?

RODA LOCAL (CPU, sem GPU/job/modelo). Precisa de pandas + numpy + a slice br_only. Nivel 1: usa os
counts CRUS do br_only + uma suposicao de cobertura de pareamento 1:1 (nao precisa do matcher).

COMPARA TRES METRICAS NO MESMO DADO SIMULADO: ΔMCC, ΔAUROC, ΔAP (average precision), todas como
diferenca-em-diferenca BR-especifica, com o MESMO bootstrap por matched-set (spec do Eduardo). O
objetivo e decidir se vale trocar o confirmatorio de ΔMCC para ΔAUROC/ΔAP.

POR QUE ISSO IMPORTA
--------------------
Com n grande e desbalanceado (br_only ~84% patogenicas), o MCC sofre: fixa UM threshold (injeta
ruido) e sua precisao e limitada pela classe menor (benignas). AUROC/AP sao threshold-free e usam
TODOS os pares pos x neg (~milhoes), entao estimam o mesmo efeito com muito mais precisao. Este
script mede isso: para o MESMO efeito verdadeiro (o shift latente que move cada metrica), qual e o
poder de detectar via IC 95% que exclui zero.

O NUMERO-CHAVE (head-to-head): no efeito verdadeiro que move o ΔMCC em 0.05, qual o poder de cada
metrica? Se poder_AUROC >> poder_MCC no MESMO dado, trocar o confirmatorio se justifica.

CAVEAT HONESTO
--------------
Este modelo aplica um efeito de DISCRIMINACAO (melhora a separacao/ranking). Um efeito puramente de
CALIBRACAO (cenario H do Eduardo: MCC/Brier mexem, AUROC/AP nao) seria invisivel ao AUROC/AP. Por
isso a recomendacao e promover AUROC/AP a confirmatorio MAS manter MCC/Brier como secundarios.
E o poder aqui e TETO: bootstrap variant-level subestima incerteza vs gene-clustered (~1.7x, v11).

USO
---
    python scripts/estimate_br_clinical_power.py \\
        --slice ~/slices/br_only.parquet --nonbr-slice ~/slices/nonbr_only.parquet \\
        --coverage 0.7 0.9 --auc 0.90 --rho 0.8 --out ~/artifacts/fase05/a1_power.json
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
log = logging.getLogger("a1_power")

_NORM = NormalDist(0.0, 1.0)
CANCER_GENES = ("BRCA1", "BRCA2", "TP53")
METRICS = ("mcc", "auroc", "ap")
DEFAULT_EFFECT_GRID = (0.0, 0.10, 0.20, 0.30, 0.50, 0.80)
# Alvos de efeito por metrica (escalas diferentes): MCC 0.05 (criterio do Eduardo); AUROC/AP 0.02
# (escala dos guardrails do Eduardo + do DiD de frequencia ~+0.027 que medimos).
METRIC_TARGET = {"mcc": 0.05, "auroc": 0.02, "ap": 0.02}


# --------------------------------------------------------------------------------------------------
# Metricas vetorizadas sobre linhas (R,n)
# --------------------------------------------------------------------------------------------------
def _mcc_rows(pred: np.ndarray, y: np.ndarray) -> np.ndarray:
    """MCC. pred:(R,n) 0/1, y:(R,n) 0/1 -> (R,)."""
    tp = (pred * y).sum(axis=1)
    fp = (pred * (1 - y)).sum(axis=1)
    fn = ((1 - pred) * y).sum(axis=1)
    tn = ((1 - pred) * (1 - y)).sum(axis=1)
    num = tp * tn - fp * fn
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return np.where(den > 0, num / den, 0.0)


def _auroc_rows(scores: np.ndarray, y: np.ndarray) -> np.ndarray:
    """AUROC via Mann-Whitney U. Scores continuos (gaussianos) => sem empates => rank ordinal exato."""
    ranks = np.argsort(np.argsort(scores, axis=1), axis=1).astype(np.float64) + 1.0
    yf = y.astype(np.float64)
    n_pos = yf.sum(axis=1)
    n_neg = scores.shape[1] - n_pos
    rank_sum_pos = (ranks * yf).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        auc = (rank_sum_pos - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg)
    return np.where((n_pos > 0) & (n_neg > 0), auc, np.nan)


def _ap_rows(scores: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Average Precision (definicao sklearn: soma de precisao nos pontos de recall)."""
    order = np.argsort(-scores, axis=1)
    sorted_y = np.take_along_axis(y.astype(np.float64), order, axis=1)
    cum_pos = np.cumsum(sorted_y, axis=1)
    positions = np.arange(1, scores.shape[1] + 1, dtype=np.float64)[None, :]
    precision = cum_pos / positions
    n_pos = sorted_y.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        ap = (precision * sorted_y).sum(axis=1) / n_pos
    return np.where(n_pos > 0, ap, np.nan)


def bc_interval(draws: np.ndarray, observed: float, alpha: float = 0.05) -> tuple[float, float]:
    """Intervalo percentil bias-corrected."""
    draws = draws[~np.isnan(draws)]
    if draws.size < 2:
        return (float("nan"), float("nan"))
    prop = float(np.mean(draws < observed))
    prop = min(max(prop, 1e-6), 1 - 1e-6)
    z0 = _NORM.inv_cdf(prop)
    lo_p = _NORM.cdf(2 * z0 + _NORM.inv_cdf(alpha / 2))
    hi_p = _NORM.cdf(2 * z0 + _NORM.inv_cdf(1 - alpha / 2))
    return (float(np.percentile(draws, 100 * lo_p)), float(np.percentile(draws, 100 * hi_p)))


# --------------------------------------------------------------------------------------------------
# Modelo generativo + DiD multi-metrica
# --------------------------------------------------------------------------------------------------
def _sim_pair_scores(y, auc, rho, rng, effect):
    """M1/M2 p/ labels y. Positivas ~N(mu,1), negativas ~N(0,1) => AUC. Ruido compartilha rho. O
    `effect` desloca as positivas no M2 (efeito de DISCRIMINACAO -- move ranking E threshold)."""
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
    mccs = _mcc_rows(preds, np.broadcast_to(y, preds.shape))
    return float(qs[int(np.argmax(mccs))])


def _did_rows(y2d, m1br, m2br, m1nb, m2nb, thr1, thr2):
    """DiD BR-especifica por metrica, sobre linhas (R,n). Retorna {metric: (R,)}."""
    out = {}

    def mcc(scores, thr):
        return _mcc_rows((scores >= thr).astype(np.int64), y2d)

    out["mcc"] = (mcc(m2br, thr2) - mcc(m1br, thr1)) - (mcc(m2nb, thr2) - mcc(m1nb, thr1))
    for name, fn in (("auroc", _auroc_rows), ("ap", _ap_rows)):
        out[name] = (fn(m2br, y2d) - fn(m1br, y2d)) - (fn(m2nb, y2d) - fn(m1nb, y2d))
    return out


def run_power_cell(n_pos, n_neg, auc, rho, effect, campaigns, boot, rng):
    """Retorna {metric: (ΔM realizado medio, poder)} p/ um efeito latente."""
    if n_pos < 2 or n_neg < 2:
        return {m: (float("nan"), float("nan")) for m in METRICS}
    y = np.concatenate([np.ones(n_pos), np.zeros(n_neg)]).astype(np.int64)
    n = len(y)
    realized = {m: [] for m in METRICS}
    detected = {m: 0 for m in METRICS}
    for _ in range(campaigns):
        m1br, m2br = _sim_pair_scores(y, auc, rho, rng, effect)  # BR recebe efeito
        m1nb, m2nb = _sim_pair_scores(y, auc, rho, rng, 0.0)     # nonBR = controle
        c1, _ = _sim_pair_scores(y, auc, rho, rng, 0.0)          # calibration p/ threshold (Eduardo)
        _, c2 = _sim_pair_scores(y, auc, rho, rng, 0.0)
        thr1, thr2 = _best_threshold(c1, y), _best_threshold(c2, y)
        pt = _did_rows(y[None], m1br[None], m2br[None], m1nb[None], m2nb[None], thr1, thr2)
        idx = rng.integers(0, n, size=(boot, n))
        draws = _did_rows(y[idx], m1br[idx], m2br[idx], m1nb[idx], m2nb[idx], thr1, thr2)
        for m in METRICS:
            obs = float(pt[m][0])
            realized[m].append(obs)
            lo, _hi = bc_interval(draws[m], obs)
            if not math.isnan(lo) and lo > 0.0:
                detected[m] += 1
    return {m: (float(np.nanmean(realized[m])), detected[m] / campaigns) for m in METRICS}


def single_mcc_ci_halfwidth(n_pos, n_neg, auc, boot, rng):
    """Self-check: meia-largura do IC 95% de UM MCC (variant-level). Anchor v11: ~0.05-0.09."""
    if n_pos < 2 or n_neg < 2:
        return float("nan")
    y = np.concatenate([np.ones(n_pos), np.zeros(n_neg)]).astype(np.int64)
    m1, _ = _sim_pair_scores(y, auc, 0.9, rng, 0.0)
    thr = _best_threshold(m1, y)
    obs = float(_mcc_rows((m1[None] >= thr).astype(np.int64), y[None])[0])
    idx = rng.integers(0, len(y), size=(boot, len(y)))
    draws = _mcc_rows((m1[idx] >= thr).astype(np.int64), y[idx])
    lo, hi = bc_interval(draws, obs)
    return (hi - lo) / 2.0


# --------------------------------------------------------------------------------------------------
# Counts
# --------------------------------------------------------------------------------------------------
def _norm_chrom_series(df):
    if "variant_key" in df.columns:
        c = df["variant_key"].astype(str).str.split(":").str[0]
    else:
        c = df["Chromosome"].astype(str).str.replace(r"\.0$", "", regex=True)
    return c.str.lower().str.replace("^chr", "", regex=True)


def count_subsets(df):
    label = pd.to_numeric(df["label"], errors="coerce")
    chrom = _norm_chrom_series(df)
    gene = df["GeneSymbol"].astype(str) if "GeneSymbol" in df.columns else pd.Series([""] * len(df))

    def counts(mask):
        sub = label[mask]
        return {"n_pos": int((sub == 1).sum()), "n_neg": int((sub == 0).sum()), "n_total": int(mask.sum())}

    out = {"all": counts(pd.Series(True, index=df.index)),
           "main_excl_chr8": counts(chrom != "8"), "chr8": counts(chrom == "8")}
    for g in CANCER_GENES:
        out[g] = counts((gene == g) & (chrom != "8"))
    return out


# --------------------------------------------------------------------------------------------------
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


def _mde_at(curve_pairs, floor):
    for dmetric, power in sorted(curve_pairs, key=lambda t: t[0]):
        if power >= floor:
            return dmetric
    return float("nan")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--slice", required=True, type=Path)
    p.add_argument("--nonbr-slice", type=Path, default=None)
    p.add_argument("--coverage", type=float, nargs="+", default=[0.7, 0.9])
    p.add_argument("--auc", type=float, nargs="+", default=[0.90])
    p.add_argument("--rho", type=float, nargs="+", default=[0.8])
    p.add_argument("--mcc-anchor", type=float, default=0.05, help="ΔMCC de referencia p/ o head-to-head")
    p.add_argument("--campaigns", type=int, default=150)
    p.add_argument("--bootstrap", type=int, default=400)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    rng = np.random.default_rng(args.seed)

    df = pd.read_parquet(args.slice)
    subsets = count_subsets(df)
    nonbr = count_subsets(pd.read_parquet(args.nonbr_slice)) if (args.nonbr_slice and args.nonbr_slice.exists()) else None

    log.info("=== COUNTS br_only ===")
    for name, c in subsets.items():
        log.info("  %-16s n_pos=%-5d n_neg=%-5d n_total=%-5d", name, c["n_pos"], c["n_neg"], c["n_total"])
    if nonbr:
        log.info("=== pool nonbr_only (contexto) ===")
        for name in ("all", "main_excl_chr8"):
            c = nonbr[name]
            log.info("  %-16s n_pos=%-5d n_neg=%-5d", name, c["n_pos"], c["n_neg"])

    main_c = subsets["main_excl_chr8"]
    hw = single_mcc_ci_halfwidth(main_c["n_pos"], main_c["n_neg"], 0.90, args.bootstrap * 4, rng)
    log.info("[self-check] meia-largura IC de UM MCC (variant-level, n=%d/%d, AUC 0.90): %.3f "
             "(anchor v11 gene-clustered ~0.05-0.09; variant-level e mais estreito, esperado)",
             main_c["n_pos"], main_c["n_neg"], hw)

    results = []
    for cov in args.coverage:
        n_pos = max(2, int(round(main_c["n_pos"] * cov)))
        n_neg = max(2, int(round(main_c["n_neg"] * cov)))
        for auc in args.auc:
            for rho in args.rho:
                curve = []
                for eff in DEFAULT_EFFECT_GRID:
                    cell = run_power_cell(n_pos, n_neg, auc, rho, eff, args.campaigns, args.bootstrap, rng)
                    curve.append({"effect_latent": eff, **{m: {"realized": cell[m][0], "power": cell[m][1]} for m in METRICS}})

                # Head-to-head: no efeito verdadeiro que move o ΔMCC em `mcc_anchor`, qual o poder
                # de cada metrica? (interpola poder_metrica vs ΔMCC realizado, avalia em mcc_anchor.)
                mcc_x = [pt["mcc"]["realized"] for pt in curve]
                head2head = {}
                for m in METRICS:
                    power_vs_mcc = list(zip(mcc_x, [pt[m]["power"] for pt in curve]))
                    head2head[m] = _interp(power_vs_mcc, args.mcc_anchor)

                # Poder no alvo proprio de cada metrica + MDE@80% (nas unidades da propria metrica).
                own = {}
                for m in METRICS:
                    pairs = [(pt[m]["realized"], pt[m]["power"]) for pt in curve]
                    own[m] = {"power_at_own_target": _interp(pairs, METRIC_TARGET[m]),
                              "target": METRIC_TARGET[m], "mde_at_80pct_power": _mde_at(pairs, 0.80)}

                results.append({"coverage": cov, "auc": auc, "rho": rho,
                                "n_pos_matched": n_pos, "n_neg_matched": n_neg,
                                "curve": curve, "head_to_head_at_mcc_anchor": head2head, "own_metric": own})
                log.info("cov=%.2f AUC=%.2f rho=%.2f | n=%d/%d | HEAD-TO-HEAD @ ΔMCC=%.2f: "
                         "poder MCC=%.2f  AUROC=%.2f  AP=%.2f",
                         cov, auc, rho, n_pos, n_neg, args.mcc_anchor,
                         head2head["mcc"], head2head["auroc"], head2head["ap"])
                for m in METRICS:
                    o = own[m]
                    log.info("    %-6s poder@Δ=%.2f: %.2f | MDE@80%%: %s", m, o["target"],
                             o["power_at_own_target"], f"{o['mde_at_80pct_power']:.3f}" if not math.isnan(o["mde_at_80pct_power"]) else "n/a")

    payload = {"slice": str(args.slice), "counts_br_only": subsets, "counts_nonbr_only": nonbr,
               "mcc_anchor": args.mcc_anchor, "metric_targets": METRIC_TARGET,
               "self_check_single_mcc_ci_halfwidth": hw, "results": results,
               "params": {"campaigns": args.campaigns, "bootstrap": args.bootstrap,
                          "coverage": args.coverage, "auc": args.auc, "rho": args.rho},
               "caveat": "Efeito modelado e de DISCRIMINACAO; efeito puro de calibracao (cenario H) "
                         "seria invisivel ao AUROC/AP -> manter MCC/Brier secundarios. Poder e TETO "
                         "(bootstrap variant-level subestima vs gene-clustered ~1.7x, v11)."}
    print(json.dumps(payload, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2))
        log.info("Escrito %s", args.out)

    _verdict(results, args.mcc_anchor)
    return 0


def _verdict(results, mcc_anchor):
    ref = [r for r in results if abs(r["coverage"] - 0.9) < 1e-9] or results
    r = ref[0]
    h = r["head_to_head_at_mcc_anchor"]
    log.info("=== VEREDITO (ref cov=%.2f, AUC=%.2f) ===", r["coverage"], r["auc"])
    log.info("No MESMO efeito verdadeiro (o que move ΔMCC=%.2f): poder MCC=%.2f vs AUROC=%.2f vs AP=%.2f (TETO)",
             mcc_anchor, h["mcc"], h["auroc"], h["ap"])
    if h["auroc"] >= 0.8 and h["auroc"] - h["mcc"] >= 0.25:
        log.info("  => TROCAR SE JUSTIFICA: AUROC (e/ou AP) detecta o mesmo efeito com poder alto onde o MCC falha.")
        log.info("     Promover ΔAUROC/ΔAP a confirmatorio; manter ΔMCC/Brier secundarios (pega efeito de calibracao).")
    elif h["auroc"] - h["mcc"] >= 0.15:
        log.info("  => AUROC/AP melhores, mas confirmar com mais campanhas/cenarios antes de bater o martelo.")
    else:
        log.info("  => Ganho de trocar e pequeno neste cenario; revisar premissas com o Eduardo.")


if __name__ == "__main__":
    raise SystemExit(main())
