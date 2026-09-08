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
    primal = d <= n
    gram = (X_tr.T @ X_tr) if primal else (X_tr @ X_tr.T)
    s, V = np.linalg.eigh(gram)
    s = np.clip(s, 0.0, None)  # eigh pode devolver -1e-12
    # Direcoes SEM suporte no treino (autovalor ~0) recebem peso zero, nao proj/(0+lambda).
    # Sem isso, ruido de ponto flutuante no espaco nulo e amplificado por 1/lambda e o resultado
    # passa a depender da implementacao de LAPACK -- medimos 0.511 numa maquina e 0.580 noutra
    # no mesmo dado. Features de posto deficiente (colunas constantes, colineares, one-hot com
    # categorias ausentes) caem nisso o tempo todo.
    keep = s > (s.max() * 1e-10 if s.size and s.max() > 0 else 0.0)
    proj = V.T @ ((X_tr.T @ yc) if primal else yc)
    proj = np.where(keep, proj, 0.0)
    for lam in lambdas:
        coef = np.where(keep, proj / (s + lam), 0.0)
        if primal:
            w = V @ coef
            out[lam] = [Xe @ w for Xe in evals]
        else:
            alpha = V @ coef
            out[lam] = [(Xe @ X_tr.T) @ alpha for Xe in evals]
    return out


def mlp_scores(X_tr, y_tr, X_val, y_val, X_test, *, seed: int, hidden: int = 64):
    """Probe nao-linear minimo. Retorna ``(scores_val, scores_test)``.

    Existe para UMA pergunta: as 68 dims das cabecas lineares estao no span do trunk, entao um probe
    LINEAR nao pode ganhar nada com elas -- se ajudarem, e por vies indutivo sob dados escassos, e so
    um modelo nao-linear revela isso. O teste ``test_mlp_probe_learns_what_ridge_cannot`` verifica
    que este probe de fato aprende o que o ridge nao aprende; sem essa verificacao ele nao serviria
    para a pergunta que motiva sua existencia.

    Early stop pela AUROC da validation. Semeado, mas NAO deterministico como o ridge -- por isso o
    ridge segue sendo o probe de ranking e este serve so a pergunta linear-vs-nao-linear.
    """
    import numpy as np
    import torch

    torch.manual_seed(seed)
    xt = torch.tensor(X_tr, dtype=torch.float32)
    yt = torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1)
    xv = torch.tensor(X_val, dtype=torch.float32)
    xe = torch.tensor(X_test, dtype=torch.float32)
    yv = np.asarray(y_val)
    net = torch.nn.Sequential(
        torch.nn.Linear(xt.shape[1], hidden), torch.nn.GELU(),
        torch.nn.Dropout(0.1), torch.nn.Linear(hidden, 1),
    )
    opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-4)
    lossf = torch.nn.BCEWithLogitsLoss()
    best, best_out, patience = -1.0, None, 0
    for epoch in range(600):
        net.train()
        opt.zero_grad()
        lossf(net(xt), yt).backward()
        opt.step()
        if epoch % 10:
            continue
        net.eval()
        with torch.no_grad():
            sv = net(xv).squeeze(1).numpy()
            se = net(xe).squeeze(1).numpy()
        score = auroc(sv[yv == 1].tolist(), sv[yv == 0].tolist()) if 0 < yv.sum() < len(yv) else 0.0
        if score > best:
            best, best_out, patience = score, (sv, se), 0
        else:
            patience += 1
            if patience >= 10:
                break
    if best_out is None:
        return np.zeros(len(X_val)), np.zeros(len(X_test))
    return best_out


# ------------------------------------------------------------------------------------------------
# avaliacao de uma configuracao
# ------------------------------------------------------------------------------------------------


def snap(scores):
    """Arredonda os scores a 1e-12 RELATIVO antes de ranquear.

    Sem isso, um probe que nao produz ordenamento nenhum (scores constantes a menos de ruido de
    ponto flutuante, ~1e-17) tem esse ruido ordenado pelo rankdata, que usa igualdade EXATA. O
    resultado e uma AUROC que passeia em torno de 0.5 em vez de ser 0.5, e que muda de maquina --
    medimos 0.511 e 0.580 no mesmo dado antes desta correcao. 1e-12 relativo esta ordens de grandeza
    abaixo de qualquer diferenca com significado e nao mascara sinal real.
    """
    scale = max((abs(v) for v in scores), default=0.0) or 1.0
    return [round(v / scale, 12) for v in scores]


def panel_auroc(scores, labels, panels) -> dict:
    """AUROC por painel, com n_P e n_B. Painel sem as duas classes fica None (indefinida)."""
    scores = snap(scores)
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
            sv, st = mlp_scores(Xtr, ytr, Xva, yva, Xte, seed=args.seed + run)
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

    used = [c for c in chosen if c is not None]
    if used:
        # So a borda SUPERIOR e sintoma: significa que o probe quis regularizar mais do que o grid
        # permite. Colar na borda inferior com d << n e o esperado -- nao ha o que regularizar.
        hi = per_run[0]["sizes"]["train"] * 1e3
        at_top = sum(1 for c in used if c >= hi * 0.99)
        if at_top:
            print(f"    [!] lambda no TETO do grid em {at_top}/{len(used)} execucoes -- "
                  "o probe queria regularizar mais; amplie a faixa")

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
