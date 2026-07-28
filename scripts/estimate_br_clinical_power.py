#!/usr/bin/env python3
"""Fase 0.5 -- sonda (a1): o endpoint clinico (DiD de MCC) tem PODER dado o n real do br_only?

RODA LOCAL (CPU, sem GPU/job/modelo). Precisa de pandas + numpy + a slice br_only. Nivel 1: usa os
counts CRUS do br_only + uma suposicao de cobertura de pareamento 1:1 -- NAO precisa do matcher
(esse e o nivel 2, pos-matcher). Assim a estimativa de poder nao trava esperando o matcher.

O QUE ESTE SCRIPT DECIDE
------------------------
O contraste confirmatorio do Eduardo e ΔMCC_BR-especifico(M2 vs M1) > 0, com IC 95% excluindo zero
e ganho >= 0.05. ΔMCC_BR-esp = [MCC(M2,BR)-MCC(M1,BR)] - [MCC(M2,nonBR)-MCC(M1,nonBR)] -- uma
dupla diferenca de MCCs sobre pares pareados. A pergunta: dado o n real do br_only (encolhido pelo
pareamento 1:1), o IC de ΔMCC e estreito o bastante pra detectar 0.05? Se nao, gastar ~9 jobs de
treino nesse endpoint e queimar a campanha -- melhor priorizar o eixo representacional/AMR e/ou o
conjunto BR-associated mais amplo (plano R03, Fase 0.5).

COMO ESTIMA
-----------
1. Conta n_pos/n_neg do br_only (total, main sem chr8, chr8, BRCA1/BRCA2/TP53) -- parte concreta.
2. Simulacao Monte Carlo do endpoint sob premissas EXPLICITAS e ajustaveis:
   - AUC base de operacao (v11 mediu br_only AUROC 0.89-0.94);
   - correlacao M1<->M2 (mesmo backbone, so o adapter muda => alta): pares mais correlacionados
     dao MENOS variancia na diferenca => MAIS poder;
   - cobertura de pareamento (quanto o 1:1 encolhe o br_only);
   - efeito verdadeiro (grid), reamostragem por MATCHED-SET (spec do Eduardo) + IC bias-corrected.
   Poder = P(IC 95% exclui 0, direcao certa).
3. Self-check: reproduz o CI de UM MCC ~±0.05-0.09 medido no v11 (gene-clustered) -- calibra o
   simulador contra a realidade conhecida.

CAVEAT IMPORTANTE (torna esta estimativa OTIMISTA)
--------------------------------------------------
O bootstrap por matched-set do Eduardo e VARIANT-LEVEL. O br_only agrupa por gene, e o v11 mostrou
que o CI GENE-CLUSTERED e ~1.7x mais largo que o variant-level. Logo o poder aqui e um TETO: o poder
real e menor. O nivel 2 (pos-matcher, com os genes reais dos pares) fecha isso.

USO
---
    python scripts/estimate_br_clinical_power.py \\
        --slice ~/slices/br_only.parquet \\
        --nonbr-slice ~/slices/nonbr_only.parquet \\
        --coverage 0.5 0.7 0.9 --auc 0.85 0.90 --rho 0.8 \\
        --out ~/artifacts/fase05/a1_power.json
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
# Grid de efeito em unidades latentes (shift nas positivas do M2 no BR); mapeado p/ ΔMCC realizado.
DEFAULT_EFFECT_GRID = (0.0, 0.15, 0.30, 0.50, 0.80, 1.20)


# --------------------------------------------------------------------------------------------------
# MCC + bootstrap bias-corrected
# --------------------------------------------------------------------------------------------------
def _mcc_rows(pred: np.ndarray, y: np.ndarray) -> np.ndarray:
    """MCC vetorizado. pred:(R,n) 0/1, y:(R,n) 0/1 (labels podem variar por linha) -> (R,)."""
    tp = (pred * y).sum(axis=1)
    fp = (pred * (1 - y)).sum(axis=1)
    fn = ((1 - pred) * y).sum(axis=1)
    tn = ((1 - pred) * (1 - y)).sum(axis=1)
    num = tp * tn - fp * fn
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return np.where(den > 0, num / den, 0.0)


def _mcc_one(pred: np.ndarray, y: np.ndarray) -> float:
    return float(_mcc_rows(pred[None, :], y[None, :])[0])


def bc_interval(draws: np.ndarray, observed: float, alpha: float = 0.05) -> tuple[float, float]:
    """Intervalo percentil bias-corrected (mesma ideia do test_abraom_vs_gnomad_power.py)."""
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
# Modelo generativo
# --------------------------------------------------------------------------------------------------
def _sim_pair_scores(
    y: np.ndarray, auc: float, rho: float, rng: np.random.Generator, effect: float
) -> tuple[np.ndarray, np.ndarray]:
    """Gera scores M1 e M2 p/ um conjunto de labels y. Positivas ~N(mu,1), negativas ~N(0,1) => AUC.
    Ruido de M1/M2 compartilha fracao rho (mesmo backbone). `effect` desloca as positivas no M2."""
    n = len(y)
    mu = math.sqrt(2.0) * _NORM.inv_cdf(auc)
    shared = rng.standard_normal(n)
    a, b = math.sqrt(rho), math.sqrt(1.0 - rho)
    base = mu * y
    m1 = base + a * shared + b * rng.standard_normal(n)
    m2 = base + a * shared + b * rng.standard_normal(n) + effect * y
    return m1, m2


def _best_threshold(scores: np.ndarray, y: np.ndarray, n_grid: int = 101) -> float:
    """Threshold que maximiza MCC (spec do Eduardo: escolhido no calibration, congelado)."""
    qs = np.quantile(scores, np.linspace(0.02, 0.98, n_grid))
    preds = (scores[None, :] >= qs[:, None]).astype(np.int64)
    mccs = _mcc_rows(preds, np.broadcast_to(y, preds.shape))
    return float(qs[int(np.argmax(mccs))])


def _dmcc_point(y, m1br, m2br, m1nb, m2nb, thr1, thr2) -> float:
    br = _mcc_one((m2br >= thr2).astype(np.int64), y) - _mcc_one((m1br >= thr1).astype(np.int64), y)
    nb = _mcc_one((m2nb >= thr2).astype(np.int64), y) - _mcc_one((m1nb >= thr1).astype(np.int64), y)
    return br - nb


def _dmcc_bootstrap(y, m1br, m2br, m1nb, m2nb, thr1, thr2, boot, rng) -> np.ndarray:
    """Bootstrap por matched-set: reamostra pares (BR_i, nonBR_i) juntos."""
    n = len(y)
    idx = rng.integers(0, n, size=(boot, n))
    yb = y[idx]
    d = (
        (_mcc_rows((m2br[idx] >= thr2).astype(np.int64), yb) - _mcc_rows((m1br[idx] >= thr1).astype(np.int64), yb))
        - (_mcc_rows((m2nb[idx] >= thr2).astype(np.int64), yb) - _mcc_rows((m1nb[idx] >= thr1).astype(np.int64), yb))
    )
    return d


def run_power_cell(n_pos, n_neg, auc, rho, effect, campaigns, boot, rng) -> tuple[float, float]:
    """Retorna (ΔMCC realizado medio, poder = P(IC 95% exclui 0, direcao positiva))."""
    if n_pos < 2 or n_neg < 2:
        return (float("nan"), float("nan"))
    y = np.concatenate([np.ones(n_pos), np.zeros(n_neg)]).astype(np.int64)
    detected, realized = 0, []
    for _ in range(campaigns):
        m1br, m2br = _sim_pair_scores(y, auc, rho, rng, effect)   # BR recebe o efeito
        m1nb, m2nb = _sim_pair_scores(y, auc, rho, rng, 0.0)      # nonBR = controle, sem efeito
        # calibration nonBR-like, sem efeito, por modelo (Eduardo: threshold do calibration set)
        c1, _ = _sim_pair_scores(y, auc, rho, rng, 0.0)
        _, c2 = _sim_pair_scores(y, auc, rho, rng, 0.0)
        thr1, thr2 = _best_threshold(c1, y), _best_threshold(c2, y)
        obs = _dmcc_point(y, m1br, m2br, m1nb, m2nb, thr1, thr2)
        lo, hi = bc_interval(_dmcc_bootstrap(y, m1br, m2br, m1nb, m2nb, thr1, thr2, boot, rng), obs)
        realized.append(obs)
        if not math.isnan(lo) and lo > 0.0:
            detected += 1
    return (float(np.mean(realized)), detected / campaigns)


def single_mcc_ci_halfwidth(n_pos, n_neg, auc, boot, rng) -> float:
    """Self-check: meia-largura do IC 95% de UM MCC (variant-level). Anchor v11: ~0.05-0.09."""
    if n_pos < 2 or n_neg < 2:
        return float("nan")
    y = np.concatenate([np.ones(n_pos), np.zeros(n_neg)]).astype(np.int64)
    m1, _ = _sim_pair_scores(y, auc, 0.9, rng, 0.0)
    thr = _best_threshold(m1, y)
    obs = _mcc_one((m1 >= thr).astype(np.int64), y)
    idx = rng.integers(0, len(y), size=(boot, len(y)))
    draws = _mcc_rows((m1[idx] >= thr).astype(np.int64), y[idx])
    lo, hi = bc_interval(draws, obs)
    return (hi - lo) / 2.0


# --------------------------------------------------------------------------------------------------
# Counts
# --------------------------------------------------------------------------------------------------
def _norm_chrom_series(df: pd.DataFrame) -> pd.Series:
    # Prefere variant_key ('8:...' string garantida); Chromosome pode vir numerico ('8.0').
    if "variant_key" in df.columns:
        c = df["variant_key"].astype(str).str.split(":").str[0]
    else:
        c = df["Chromosome"].astype(str).str.replace(r"\.0$", "", regex=True)
    return c.str.lower().str.replace("^chr", "", regex=True)


def count_subsets(df: pd.DataFrame) -> dict[str, dict[str, int]]:
    label = pd.to_numeric(df["label"], errors="coerce")
    chrom = _norm_chrom_series(df)
    gene = df["GeneSymbol"].astype(str) if "GeneSymbol" in df.columns else pd.Series([""] * len(df))

    def counts(mask: pd.Series) -> dict[str, int]:
        sub = label[mask]
        return {"n_pos": int((sub == 1).sum()), "n_neg": int((sub == 0).sum()), "n_total": int(mask.sum())}

    out = {
        "all": counts(pd.Series(True, index=df.index)),
        "main_excl_chr8": counts(chrom != "8"),
        "chr8": counts(chrom == "8"),
    }
    for g in CANCER_GENES:
        out[g] = counts((gene == g) & (chrom != "8"))
    return out


# --------------------------------------------------------------------------------------------------
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--slice", required=True, type=Path, help="parquet do br_only")
    p.add_argument("--nonbr-slice", type=Path, default=None, help="parquet do nonbr_only (pool de pareamento, contexto)")
    p.add_argument("--coverage", type=float, nargs="+", default=[0.5, 0.7, 0.9], help="cenarios de cobertura do pareamento 1:1")
    p.add_argument("--auc", type=float, nargs="+", default=[0.85, 0.90], help="AUC base de operacao")
    p.add_argument("--rho", type=float, nargs="+", default=[0.8], help="correlacao M1<->M2")
    p.add_argument("--target", type=float, default=0.05, help="ΔMCC alvo do criterio de sucesso")
    p.add_argument("--campaigns", type=int, default=200, help="reps Monte Carlo (poder)")
    p.add_argument("--bootstrap", type=int, default=500, help="reamostras matched-set por campanha")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    rng = np.random.default_rng(args.seed)

    df = pd.read_parquet(args.slice)
    subsets = count_subsets(df)
    nonbr_subsets = None
    if args.nonbr_slice and args.nonbr_slice.exists():
        nonbr_subsets = count_subsets(pd.read_parquet(args.nonbr_slice))

    log.info("=== COUNTS br_only ===")
    for name, c in subsets.items():
        log.info("  %-16s n_pos=%-5d n_neg=%-5d n_total=%-5d", name, c["n_pos"], c["n_neg"], c["n_total"])
    if nonbr_subsets:
        log.info("=== pool nonbr_only (contexto de cobertura) ===")
        for name in ("all", "main_excl_chr8"):
            c = nonbr_subsets[name]
            log.info("  %-16s n_pos=%-5d n_neg=%-5d", name, c["n_pos"], c["n_neg"])

    # Self-check do simulador contra o anchor v11 (~0.05-0.09 p/ UM MCC no n do br_only main).
    main = subsets["main_excl_chr8"]
    hw = single_mcc_ci_halfwidth(main["n_pos"], main["n_neg"], 0.90, args.bootstrap * 4, rng)
    log.info("[self-check] meia-largura IC de UM MCC (variant-level, n=%d/%d, AUC 0.90): %.3f "
             "(anchor v11 gene-clustered ~0.05-0.09; variant-level e mais estreito, esperado)",
             main["n_pos"], main["n_neg"], hw)

    # Grade de poder sobre o br_only MAIN (endpoint confirmatorio), por cobertura/AUC/rho/efeito.
    power_results = []
    for cov in args.coverage:
        n_pos = max(2, int(round(main["n_pos"] * cov)))
        n_neg = max(2, int(round(main["n_neg"] * cov)))
        for auc in args.auc:
            for rho in args.rho:
                curve = []
                for eff in DEFAULT_EFFECT_GRID:
                    realized, power = run_power_cell(n_pos, n_neg, auc, rho, eff, args.campaigns, args.bootstrap, rng)
                    curve.append({"effect_latent": eff, "dmcc_realized": realized, "power": power})
                # poder no ΔMCC alvo (interpola pela curva realized->power) e MDE @80%
                valid = [pt for pt in curve if not math.isnan(pt["dmcc_realized"])]
                power_at_target = _interp_power(valid, args.target)
                mde80 = _mde_at(valid, 0.80)
                power_results.append({
                    "coverage": cov, "auc": auc, "rho": rho,
                    "n_pos_matched": n_pos, "n_neg_matched": n_neg,
                    "curve": curve, "power_at_target": power_at_target, "mde_at_80pct_power": mde80,
                })
                log.info("cov=%.2f AUC=%.2f rho=%.2f | n=%d/%d | poder@ΔMCC=%.2f: %.2f | MDE@80%%: %s",
                         cov, auc, rho, n_pos, n_neg, args.target, power_at_target,
                         f"{mde80:.3f}" if not math.isnan(mde80) else "nao alcancado")

    payload = {
        "slice": str(args.slice), "counts_br_only": subsets,
        "counts_nonbr_only": nonbr_subsets, "target_dmcc": args.target,
        "self_check_single_mcc_ci_halfwidth": hw, "power": power_results,
        "params": {"campaigns": args.campaigns, "bootstrap": args.bootstrap,
                   "coverage": args.coverage, "auc": args.auc, "rho": args.rho},
        "caveat": "Poder e TETO: bootstrap variant-level (spec Eduardo) subestima incerteza vs "
                  "gene-clustered (v11: ~1.7x mais largo). Poder real e menor. chr8/genes: se "
                  "n_pos<20 ou n_neg<20 => descritivo (regra do Eduardo).",
    }
    print(json.dumps(payload, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2))
        log.info("Escrito %s", args.out)

    _print_verdict(power_results, args.target)
    return 0


def _interp_power(curve, target_dmcc) -> float:
    """Interpola poder no ΔMCC alvo pela curva (dmcc_realized crescente -> power)."""
    pts = sorted(((pt["dmcc_realized"], pt["power"]) for pt in curve), key=lambda t: t[0])
    if not pts:
        return float("nan")
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    if target_dmcc <= xs[0]:
        return ys[0]
    if target_dmcc >= xs[-1]:
        return ys[-1]
    return float(np.interp(target_dmcc, xs, ys))


def _mde_at(curve, power_floor) -> float:
    """Menor ΔMCC realizado com poder >= floor (efeito minimo detectavel)."""
    pts = sorted(((pt["dmcc_realized"], pt["power"]) for pt in curve), key=lambda t: t[0])
    for dmcc, power in pts:
        if power >= power_floor:
            return dmcc
    return float("nan")


def _print_verdict(power_results, target) -> None:
    # cenario de referencia: cobertura 0.7, AUC 0.90, primeiro rho
    ref = [r for r in power_results if abs(r["coverage"] - 0.7) < 1e-9 and abs(r["auc"] - 0.90) < 1e-9]
    if not ref:
        ref = power_results
    p = ref[0]["power_at_target"] if ref else float("nan")
    log.info("=== VEREDITO (ref: cov=0.70, AUC=0.90) ===")
    log.info("Poder p/ detectar ΔMCC_BR-esp=%.2f: %.2f (TETO -- real e menor por gene-clustering)", target, p)
    if math.isnan(p):
        log.info("  n insuficiente pra estimar.")
    elif p < 0.5:
        log.info("  SUB-POTENTE: endpoint clinico MCC nao e viavel como confirmatorio no n atual. "
                 "Priorizar eixo representacional/AMR e/ou conjunto BR-associated mais amplo. Levar ao Eduardo.")
    elif p < 0.8:
        log.info("  MARGINAL: viavel so sob premissas otimistas (AUC/rho altos, cobertura alta). Decidir com o Eduardo.")
    else:
        log.info("  POTENTE (teto): o endpoint pode ter poder; confirmar no nivel 2 (pos-matcher, gene-clustered).")


if __name__ == "__main__":
    raise SystemExit(main())
