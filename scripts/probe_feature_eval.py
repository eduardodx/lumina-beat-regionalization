#!/usr/bin/env python3
"""R0 -- harness de avaliacao de features sob o protocolo do Mosaic. CPU, roda em qualquer lugar.

E o instrumento de medida das ablacoes: dado um conjunto de blocos de features, responde "quao boa
e esta representacao?" seguindo o protocolo do release em vez de uma validacao qualquer.

POR QUE RIDGE E NAO REGRESSAO LOGISTICA
---------------------------------------
Vamos comparar dezenas de configuracoes, incluindo 27 camadas entre si. Um otimizador iterativo
injeta variancia (inicializacao, numero de passos, tolerancia) que pode ser MAIOR que a diferenca
entre camadas -- o ranking viraria ruido. Ridge tem solucao fechada: e deterministico, e uma unica
decomposicao espectral do Gram entrega TODOS os lambdas de graca. Para AUROC, que so depende do
ordenamento, ridge sobre rotulos centrados e um ranker perfeitamente adequado.

O probe MLP (--mlp) existe para uma pergunta especifica: as 68 dimensoes das cabecas lineares estao
no span do trunk, entao um probe LINEAR nao pode ganhar nada com elas. Se elas ajudarem, tem que ser
por viés indutivo sob dados escassos -- e so um probe nao-linear (ou regularizado de outro jeito)
revela isso. A diferenca entre os dois probes E o experimento.

PROTOCOLO (ver eval/embedding_probe/protocol.py para as regras literais)
    teste = fold i (gold) · validation = fold (i+1)%5 (gold) · treino = os outros 3
    selecao de lambda: macro AUROC NAO ponderada de missense/splice/noncoding na VALIDATION
    agregacao entre folds: ponderada por n_P x n_B
    padronizacao ajustada SO no treino; o teste nunca participa de nenhuma escolha

USO
    PYTHONPATH="$WORK" "$PY" scripts/probe_feature_eval.py \\
        --features ~/probe/rich/probe_features.npz \\
        --release-root ~/mosaic-v1 --configs configs/probe_feature_configs.json \\
        --out ~/probe/rich/feature_eval.json
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

from eval.embedding_probe.protocol import (  # noqa: E402
    DISCRIMINATION_PANELS,
    K_FOLDS,
    TRACKS,
    assert_no_unit_leak,
    cross_fitted_split,
    macro_auroc,
    weighted_cross_fold,
    worst_panel,
)
from eval.embedding_probe.stats import auroc  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--features", type=Path, required=True, help="npz com um array por bloco + variant_id")
    p.add_argument("--release-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--configs", type=Path, default=None,
                   help="JSON {nome: [blocos]}; omitido = um config por bloco + 'todos'")
    p.add_argument("--track", default="core_locus", choices=sorted(TRACKS))
    p.add_argument("--train-tiers", default="gold", choices=["gold", "gold+consensus"],
                   help="'gold' e o desvio barato da fase de ablacao; o contrato e gold+consensus")
    p.add_argument("--mlp", action="store_true", help="roda tambem um probe MLP (precisa de torch)")
    p.add_argument("--n-lambda", type=int, default=9)
    p.add_argument("--seed", type=int, default=20260908)
    return p.parse_args(argv)


# ------------------------------------------------------------------------------------------------
# probe ridge: uma decomposicao, todos os lambdas
# ------------------------------------------------------------------------------------------------


def ridge_scores(X_tr, y_tr, evals, lambdas):
    """Scores de ridge para cada lambda. Uma decomposicao espectral serve todos.

    Usa o Gram menor: primal (d x d) quando d <= n, dual (n x n) caso contrario -- assim a mesma
    rotina serve tanto um bloco de 384 dims quanto a concatenacao de todos.
    """
    import numpy as np

    n, d = X_tr.shape
    yc = y_tr - y_tr.mean()
    out: dict[float, list] = {}
    if d <= n:
        gram = X_tr.T @ X_tr
        s, V = np.linalg.eigh(gram)
        s = np.clip(s, 0.0, None)  # eigh pode devolver -1e-12
        proj = V.T @ (X_tr.T @ yc)
        for lam in lambdas:
            w = V @ (proj / (s + lam))
            out[lam] = [Xe @ w for Xe in evals]
    else:
        gram = X_tr @ X_tr.T
        s, V = np.linalg.eigh(gram)
        s = np.clip(s, 0.0, None)
        proj = V.T @ yc
        for lam in lambdas:
            alpha = V @ (proj / (s + lam))
            out[lam] = [(Xe @ X_tr.T) @ alpha for Xe in evals]
    return out


def mlp_scores(X_tr, y_tr, evals, *, seed: int, hidden: int = 64):
    """Probe nao-linear minimo. Early stop pela macro da validation (evals[0])."""
    import numpy as np
    import torch

    torch.manual_seed(seed)
    xt = torch.tensor(X_tr, dtype=torch.float32)
    yt = torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1)
    net = torch.nn.Sequential(
        torch.nn.Linear(xt.shape[1], hidden), torch.nn.GELU(),
        torch.nn.Dropout(0.2), torch.nn.Linear(hidden, 1),
    )
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = torch.nn.BCEWithLogitsLoss()
    ev = [torch.tensor(e, dtype=torch.float32) for e in evals]
    best, best_scores, patience = -1.0, None, 0
    for epoch in range(200):
        net.train()
        opt.zero_grad()
        lossf(net(xt), yt).backward()
        opt.step()
        if epoch % 5 == 0:
            net.eval()
            with torch.no_grad():
                scores = [net(e).squeeze(1).numpy() for e in ev]
            # criterio de parada: AUROC global na validation (proxy barato da macro)
            v = scores[0]
            yv = getattr(mlp_scores, "_yval", None)
            score = 0.0 if yv is None else auroc(v[yv == 1].tolist(), v[yv == 0].tolist())
            if score > best:
                best, best_scores, patience = score, scores, 0
            else:
                patience += 1
                if patience >= 6:
                    break
    return best_scores if best_scores is not None else [np.zeros(len(e)) for e in evals]


# ------------------------------------------------------------------------------------------------
# avaliacao de uma configuracao
# ------------------------------------------------------------------------------------------------


def panel_auroc(scores, labels, panels) -> dict:
    """AUROC por painel, com n_P e n_B. Painel sem as duas classes fica None (indefinida)."""
    out = {}
    for panel in DISCRIMINATION_PANELS:
        mask = [i for i, p in enumerate(panels) if p == panel]
        pos = [scores[i] for i in mask if labels[i] == 1]
        neg = [scores[i] for i in mask if labels[i] == 0]
        out[panel] = {
            "auroc": auroc(pos, neg) if pos and neg else None,
            "n_pos": len(pos), "n_neg": len(neg),
        }
    return out


def evaluate_config(X, meta, blocks, args, *, use_mlp=False) -> dict:
    import numpy as np

    folds, tiers, units = meta["folds"], meta["tiers"], meta["units"]
    labels, panels = meta["labels"], meta["panels"]
    train_tiers = ("gold",) if args.train_tiers == "gold" else ("gold", "consensus")
    per_run, chosen = [], []

    for run in range(K_FOLDS):
        split = cross_fitted_split(folds, tiers, run_id=run, train_tiers=train_tiers)
        assert_no_unit_leak(split, units)
        if not split.train or not split.validation or not split.test:
            continue
        tr, va, te = np.array(split.train), np.array(split.validation), np.array(split.test)
        Xtr = X[tr]
        mu, sd = Xtr.mean(0), Xtr.std(0)
        sd[sd < 1e-8] = 1.0  # dimensao constante no treino: nao escala, nao explode
        Xtr = (Xtr - mu) / sd
        Xva, Xte = (X[va] - mu) / sd, (X[te] - mu) / sd
        ytr = np.array([labels[i] for i in tr], dtype=np.float64)
        yva = np.array([labels[i] for i in va])
        yte = [labels[i] for i in te]
        pva = [panels[i] for i in va]
        pte = [panels[i] for i in te]

        if use_mlp:
            mlp_scores._yval = yva  # criterio de early stop
            sv, st = mlp_scores(Xtr, ytr, [Xva, Xte], seed=args.seed + run)
            best = {"lambda": None, "val_macro": None, "val": sv, "test": st}
        else:
            lambdas = (len(tr) * np.logspace(-5, 3, args.n_lambda)).tolist()
            paths = ridge_scores(Xtr, ytr, [Xva, Xte], lambdas)
            best = None
            for lam, (sv, st) in paths.items():
                m = macro_auroc({k: v["auroc"] for k, v in
                                 panel_auroc(sv.tolist(), yva.tolist(), pva).items()})
                if m is not None and (best is None or m > best["val_macro"]):
                    best = {"lambda": lam, "val_macro": m, "val": sv, "test": st}
            if best is None:  # nenhum lambda deu macro completa na validation
                continue

        test_panels = panel_auroc(best["test"].tolist(), yte, pte)
        per_run.append({"run_id": run, "lambda": best["lambda"], "val_macro": best["val_macro"],
                        "sizes": split.sizes, "panels": test_panels})
        chosen.append(best["lambda"])

    if not per_run:
        return {"blocks": blocks, "n_runs": 0, "error": "nenhuma execucao avaliavel"}

    agg = {}
    for panel in DISCRIMINATION_PANELS:
        agg[panel] = weighted_cross_fold(
            [(r["panels"][panel]["auroc"], r["panels"][panel]["n_pos"], r["panels"][panel]["n_neg"])
             for r in per_run if r["panels"][panel]["auroc"] is not None]
        )
    return {
        "blocks": blocks, "n_dims": int(X.shape[1]), "n_runs": len(per_run),
        "per_panel": agg, "macro": macro_auroc(agg), "worst_panel": worst_panel(agg),
        "lambdas_chosen": chosen, "per_run": per_run,
    }


# ------------------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    import numpy as np
    import pandas as pd

    with np.load(args.features, allow_pickle=False) as z:
        block_names = [k for k in z.files if k.startswith("blk_")]
        if not block_names:
            raise SystemExit(f"nenhum array 'blk_*' em {args.features}; achei {z.files}")
        vid = [str(v) for v in z["variant_id"]]
        raw_blocks = {b: z[b] for b in block_names}  # materializa antes de fechar o handle
    print(f"[eval] {len(vid):,} variantes · {len(block_names)} blocos: "
          + ", ".join(b[4:] for b in sorted(block_names)[:8])
          + (" ..." if len(block_names) > 8 else ""))

    ex = pd.read_parquet(args.release_root / "pb_examples.parquet",
                         columns=["variant_id", "binary_label", "label_tier"])
    pa_ = pd.read_parquet(args.release_root / "pb_panels.parquet",
                          columns=["variant_id", "primary_panel"])
    fold_col, unit_col = TRACKS[args.track]
    pt = pd.read_parquet(args.release_root / "pb_partitions.parquet",
                         columns=["variant_id", fold_col, unit_col])
    rel = ex.merge(pa_, on="variant_id").merge(pt, on="variant_id").set_index("variant_id")

    keep = [i for i, v in enumerate(vid) if v in rel.index]
    if len(keep) < len(vid):
        print(f"[eval] {len(vid) - len(keep):,} variantes fora do release, descartadas")
    sub = rel.loc[[vid[i] for i in keep]]
    meta = {
        "folds": [None if pd.isna(f) else int(f) for f in sub[fold_col]],
        "tiers": sub["label_tier"].tolist(),
        "units": sub[unit_col].astype(str).tolist(),
        "labels": [int(v) for v in sub["binary_label"]],
        "panels": sub["primary_panel"].tolist(),
    }
    n_gold = sum(1 for t in meta["tiers"] if t == "gold")
    print(f"[eval] track={args.track} · treino={args.train_tiers} · {n_gold:,} gold de "
          f"{len(keep):,} usaveis")

    blocks = {b[4:]: raw_blocks[b][keep].astype(np.float64) for b in block_names}
    if args.configs:
        configs = json.loads(args.configs.read_text(encoding="utf-8"))
    else:
        configs = {name: [name] for name in sorted(blocks)}
        configs["__todos__"] = sorted(blocks)

    report = {"features": str(args.features), "track": args.track,
              "train_tiers": args.train_tiers, "n_variants": len(keep), "n_gold": n_gold,
              "probe": "mlp" if args.mlp else "ridge", "configs": {}}
    print(f"\n{'configuracao':<28} {'dims':>6} {'macro':>7} {'pior':>7} "
          f"{'missense':>9} {'splice':>7} {'noncoding':>10}")
    print("-" * 82)
    for name, wanted in configs.items():
        missing = [b for b in wanted if b not in blocks]
        if missing:
            print(f"  {name}: blocos ausentes {missing}, pulando")
            continue
        t0 = time.perf_counter()
        X = np.concatenate([blocks[b] for b in wanted], axis=1)
        res = evaluate_config(X, meta, wanted, args, use_mlp=args.mlp)
        res["seconds"] = time.perf_counter() - t0
        report["configs"][name] = res
        if res.get("n_runs"):
            f = lambda v: f"{v:.4f}" if v is not None else "   --"  # noqa: E731
            p = res["per_panel"]
            print(f"{name:<28} {res['n_dims']:>6} {f(res['macro']):>7} {f(res['worst_panel']):>7} "
                  f"{f(p['missense']):>9} {f(p['splice']):>7} {f(p['noncoding']):>10}")
        else:
            print(f"{name:<28} {'--':>6}  {res.get('error')}")

    ranked = sorted(((r["macro"], n) for n, r in report["configs"].items() if r.get("macro")),
                    reverse=True)
    if ranked:
        print(f"\n[eval] melhor: {ranked[0][1]} (macro {ranked[0][0]:.4f})")
        report["ranking"] = [{"config": n, "macro": m} for m, n in ranked]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n",
                        encoding="utf-8")
    print(f"[eval] -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
