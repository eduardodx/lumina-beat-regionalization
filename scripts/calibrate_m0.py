#!/usr/bin/env python3
"""Platt scaling + threshold de MCC para a campanha R03 (item (a) da Fase 1).

Implementa o procedimento de calibracao do documento do Eduardo:
  * §9.4 — Platt scaling: p_cal = sigma(a * s + b), com (a, b) AJUSTADOS NO CALIBRATION set.
    Brier so deve ser calculado sobre probabilidades calibradas (nao sobre logits crus).
  * §9.1 — threshold: escolher o limiar que MAXIMIZA MCC no calibration set, CONGELAR, e aplicar
    exatamente o mesmo em T_BR e T_nonBR (nunca escolher limiares separados por conjunto).

Fluxo (dois subcomandos):
  fit   -> le o test_predictions.parquet do M0 (predicoes CRUAS no calibration: colunas `label` +
           `probability`), ajusta (a, b) por Newton, escolhe tau que maximiza MCC nas probs
           CALIBRADAS, e salva calibration_m0.json = {platt_a, platt_b, threshold, metrics...}.
  apply -> le um predictions.parquet qualquer (ex.: T_BR / T_nonBR) + o calibration_m0.json, aplica
           (a, b, tau) CONGELADOS e reporta as metricas calibradas. E este passo que produz a foto
           do M0 em T_BR e T_nonBR (Fase 4 / Tabela 3).

Observacoes de rigor:
  * AUROC e AP (average precision) sao invariantes a transformacao monotona -> iguais em prob crua ou
    calibrada; o que a calibracao move e Brier / log-loss / MCC@tau (cenarios H e I do §15).
  * O parquet so guarda `probability` (= sigmoid do logit do modelo); recuperamos o score bruto
    s = logit(probability) pela inversa da sigmoid (com clip), que e exatamente o logit original.
  * Zero dependencias alem de numpy/pandas: as metricas reusam eval.clinvar.metrics (numpy puro).

Rodar (no notebook, com o env do R03):
    cd "$WORK" && PYTHONPATH="$WORK" "$PY" scripts/calibrate_m0.py fit \
        --predictions ~/artifacts/fase1/m0_full/test_predictions.parquet \
        --out ~/artifacts/fase1/m0_full/calibration_m0.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.clinvar.metrics import (  # noqa: E402
    binary_auprc,
    binary_log_loss,
    binary_roc_auc,
    brier_score,
    classification_metrics,
    optimize_threshold,
)

_EPS = 1e-6


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Sigmoid numericamente estavel (sem overflow em |z| grande)."""
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def _logit(p: np.ndarray) -> np.ndarray:
    """Inversa da sigmoid: recupera o score bruto s a partir da probabilidade salva."""
    p = np.clip(np.asarray(p, dtype=np.float64), _EPS, 1.0 - _EPS)
    return np.log(p / (1.0 - p))


def fit_platt(s: np.ndarray, y: np.ndarray, *, max_iter: int = 100, tol: float = 1e-9,
              ridge: float = 1e-10) -> tuple[float, float]:
    """Ajusta p = sigma(a*s + b) por Newton-Raphson (2 parametros). Inicio em (a=1, b=0) = identidade
    (recupera as probs originais). Converge em ~5-10 iteracoes; ridge minimo estabiliza a Hessiana."""
    s = np.asarray(s, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    a, b = 1.0, 0.0
    for _ in range(max_iter):
        p = _sigmoid(a * s + b)
        w = np.clip(p * (1.0 - p), 1e-12, None)  # peso IRLS
        resid = p - y
        ga = float(np.sum(resid * s))
        gb = float(np.sum(resid))
        haa = float(np.sum(w * s * s)) + ridge
        hab = float(np.sum(w * s))
        hbb = float(np.sum(w)) + ridge
        det = haa * hbb - hab * hab
        if abs(det) < 1e-18:
            break
        da = (hbb * ga - hab * gb) / det
        db = (-hab * ga + haa * gb) / det
        a -= da
        b -= db
        if max(abs(da), abs(db)) < tol:
            break
    return float(a), float(b)


def apply_platt(probs: np.ndarray, a: float, b: float) -> np.ndarray:
    return _sigmoid(a * _logit(probs))  # sigma(a*logit(p) + b) ; b entra via _sigmoid


def _metrics_block(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, Any]:
    y = np.asarray(y, dtype=np.int64)
    p = np.asarray(p, dtype=np.float64)
    at_thr = classification_metrics(y, p, threshold)
    return {
        "n": int(len(y)),
        "n_positive": int((y == 1).sum()),
        "n_negative": int((y == 0).sum()),
        "prevalence": float(np.mean(y == 1)) if len(y) else float("nan"),
        "auroc": binary_roc_auc(y, p),
        "auprc": binary_auprc(y, p),
        "brier_score": brier_score(y, p),
        "log_loss": binary_log_loss(y, p),
        "mcc": at_thr["mcc"],
        "f1": at_thr["f1"],
        "precision": at_thr["precision"],
        "recall": at_thr["recall"],
        "specificity": at_thr["specificity"],
        "tp": at_thr["tp"], "tn": at_thr["tn"], "fp": at_thr["fp"], "fn": at_thr["fn"],
        "threshold": float(threshold),
    }


def _read(predictions: Path, label_col: str, prob_col: str) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_parquet(predictions)
    for col in (label_col, prob_col):
        if col not in df.columns:
            raise KeyError(f"Coluna {col!r} ausente em {predictions} (colunas: {list(df.columns)}).")
    y = df[label_col].to_numpy().astype(np.int64)
    p = df[prob_col].to_numpy().astype(np.float64)
    return y, p


def cmd_fit(args: argparse.Namespace) -> int:
    y, p_raw = _read(args.predictions, args.label_col, args.prob_col)
    s = _logit(p_raw)
    a, b = fit_platt(s, y)
    p_cal = _sigmoid(a * s + b)
    tau, _ = optimize_threshold(y, p_cal, "mcc")  # §9.1: limiar que maximiza MCC no calibration

    result = {
        "platt_a": a,
        "platt_b": b,
        "threshold": float(tau),
        "threshold_metric": "mcc",
        "threshold_source": "calibration_set (Eduardo §9.1)",
        "calibration_predictions": str(args.predictions),
        "n_calibration": int(len(y)),
        "metrics_raw_at_0.5": _metrics_block(y, p_raw, 0.5),
        "metrics_calibrated_at_threshold": _metrics_block(y, p_cal, tau),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    m = result["metrics_calibrated_at_threshold"]
    print(f"platt_a={a:.6f} platt_b={b:.6f} threshold(tau)={tau:.6f}")
    print(f"[calibration] MCC={m['mcc']:.4f} AUROC={m['auroc']:.4f} AP={m['auprc']:.4f} "
          f"Brier={m['brier_score']:.4f} (raw Brier@0.5={result['metrics_raw_at_0.5']['brier_score']:.4f})")
    print(f"saved={args.out}")
    return 0


def cmd_apply(args: argparse.Namespace) -> int:
    cal = json.loads(args.calibration.read_text(encoding="utf-8"))
    a, b, tau = float(cal["platt_a"]), float(cal["platt_b"]), float(cal["threshold"])
    y, p_raw = _read(args.predictions, args.label_col, args.prob_col)
    p_cal = _sigmoid(a * _logit(p_raw) + b)

    block = {
        "predictions": str(args.predictions),
        "calibration": str(args.calibration),
        "platt_a": a, "platt_b": b, "threshold": tau,
        "metrics_raw_at_threshold": _metrics_block(y, p_raw, tau),
        "metrics_calibrated_at_threshold": _metrics_block(y, p_cal, tau),
    }
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(block, indent=2), encoding="utf-8")
    if args.save_calibrated_parquet:
        df = pd.read_parquet(args.predictions)
        df["probability_calibrated"] = p_cal
        df["prediction_at_frozen_threshold"] = (p_cal >= tau).astype(int)
        df.to_parquet(args.predictions, index=False)

    mc = block["metrics_calibrated_at_threshold"]
    print(f"[{args.predictions.name}] n={mc['n']} pos={mc['n_positive']} neg={mc['n_negative']} "
          f"prev={mc['prevalence']:.3f}")
    print(f"  calibrado @tau={tau:.4f}: MCC={mc['mcc']:.4f} AUROC={mc['auroc']:.4f} "
          f"AP={mc['auprc']:.4f} Brier={mc['brier_score']:.4f}")
    if args.out is not None:
        print(f"  saved={args.out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--prob-col", default="probability")
    sub = parser.add_subparsers(dest="command", required=True)

    p_fit = sub.add_parser("fit", help="ajusta Platt + threshold no calibration set")
    p_fit.add_argument("--predictions", type=Path, required=True, help="test_predictions.parquet do M0")
    p_fit.add_argument("--out", type=Path, required=True, help="calibration_m0.json de saida")
    p_fit.set_defaults(func=cmd_fit)

    p_app = sub.add_parser("apply", help="aplica (a,b,tau) congelados a um predictions.parquet")
    p_app.add_argument("--predictions", type=Path, required=True)
    p_app.add_argument("--calibration", type=Path, required=True, help="calibration_m0.json do fit")
    p_app.add_argument("--out", type=Path, default=None, help="json de metricas (opcional)")
    p_app.add_argument("--save-calibrated-parquet", action="store_true",
                       help="grava probability_calibrated de volta no parquet de predicoes")
    p_app.set_defaults(func=cmd_apply)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
