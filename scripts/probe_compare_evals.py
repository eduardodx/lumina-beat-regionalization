#!/usr/bin/env python3
"""Compara duas avaliacoes do probe (JSON de probe_feature_eval.py), configuracao a configuracao.

Existe para uma pergunta: as conclusoes da pesquisa de extracao resistem quando o probe MLP passa a
escolher a epoca pelo MESMO criterio do ridge (macro dos paineis de discriminacao)? Le a avaliacao
antiga e a nova e mostra, por configuracao, a macro e o missense antes e depois, e a mudanca de posicao
no ranking.

``--contrast A:B`` refaz as comparacoes em que as conclusoes se apoiaram: a diferenca A - B na
avaliacao antiga, na nova, e se o SINAL se manteve, conferido separadamente na macro e no missense
(uma diferenca pode se manter na macro e inverter no missense). Uma conclusao que troca de sinal nao
sobreviveu; diferenca exatamente zero nao tem sinal e aparece como ``--``.

USO
---
    python scripts/probe_compare_evals.py ANTIGO.json NOVO.json \\
        --contrast so_cabecas_lineares:so_hidden --contrast max_mais_focal:so_hidden
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PANEL = "missense"
METRICS = ("macro", PANEL)
FLAG = {True: "mantido", False: "INVERTIDO", None: "--"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("old", type=Path, help="avaliacao antiga (JSON)")
    p.add_argument("new", type=Path, help="avaliacao nova (JSON)")
    p.add_argument("--contrast", action="append", default=[], metavar="A:B",
                   help="diferenca A - B a comparar entre as duas avaliacoes (repetivel)")
    p.add_argument("--top", type=int, default=15, help="quantas maiores mudancas de macro listar")
    return p.parse_args(argv)


def load_configs(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")).get("configs", {})


def value(cfg: dict | None, key: str):
    """``macro`` ou a AUROC agregada de um painel (``per_panel[painel]`` e um float)."""
    if not cfg:
        return None
    if key == "macro":
        return cfg.get("macro")
    return (cfg.get("per_panel") or {}).get(key)


def delta(a, b):
    return (a - b) if isinstance(a, (int, float)) and isinstance(b, (int, float)) else None


def sign_kept(old, new):
    """Se a diferenca manteve o sinal; ``None`` quando falta um lado ou um deles e exatamente zero."""
    if not isinstance(old, (int, float)) or not isinstance(new, (int, float)) or old == 0 or new == 0:
        return None
    return (old > 0) == (new > 0)


def ranks(configs: dict) -> dict:
    ordered = sorted((c for c in configs if configs[c].get("macro") is not None),
                     key=lambda c: -configs[c]["macro"])
    return {c: i + 1 for i, c in enumerate(ordered)}


def compare(old: dict, new: dict, contrasts: list[str]) -> dict:
    common = sorted(set(old) & set(new))
    r_old, r_new = ranks(old), ranks(new)
    rows = []
    for name in common:
        row = {
            "config": name,
            "macro_old": value(old[name], "macro"), "macro_new": value(new[name], "macro"),
            "missense_old": value(old[name], PANEL), "missense_new": value(new[name], PANEL),
            "rank_old": r_old.get(name), "rank_new": r_new.get(name),
        }
        row["d_macro"] = delta(row["macro_new"], row["macro_old"])
        row["d_missense"] = delta(row["missense_new"], row["missense_old"])
        rows.append(row)

    out = []
    for spec in contrasts:
        a, _, b = spec.partition(":")
        res: dict = {"contrast": spec}
        for label, src in (("old", old), ("new", new)):
            for metric in METRICS:
                res[f"{metric}_{label}"] = delta(value(src.get(a), metric), value(src.get(b), metric))
        res["sinal"] = {metric: sign_kept(res[f"{metric}_old"], res[f"{metric}_new"]) for metric in METRICS}
        out.append(res)
    return {"rows": rows, "contrasts": out,
            "so_no_antigo": sorted(set(old) - set(new)), "so_no_novo": sorted(set(new) - set(old))}


def _f(v) -> str:
    return f"{v:.4f}" if isinstance(v, (int, float)) else "--"


def _d(v) -> str:
    return f"{v:+.4f}" if isinstance(v, (int, float)) else "--"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    res = compare(load_configs(args.old), load_configs(args.new), args.contrast)
    rows = sorted((r for r in res["rows"] if r["d_macro"] is not None), key=lambda r: -abs(r["d_macro"]))

    print(f"[compare] {len(res['rows'])} configuracoes em comum · antigo={args.old.name} · novo={args.new.name}")
    if res["so_no_antigo"] or res["so_no_novo"]:
        print(f"[compare] so no antigo: {res['so_no_antigo']} · so no novo: {res['so_no_novo']}")

    print(f"\n{'configuracao':<36}{'macro ant':>10}{'macro nov':>10}{'delta':>9}"
          f"{'miss ant':>10}{'miss nov':>10}{'delta':>9}{'rank':>10}")
    for r in rows[: args.top]:
        rank = f"{r['rank_old']}->{r['rank_new']}"
        print(f"{r['config']:<36}{_f(r['macro_old']):>10}{_f(r['macro_new']):>10}{_d(r['d_macro']):>9}"
              f"{_f(r['missense_old']):>10}{_f(r['missense_new']):>10}{_d(r['d_missense']):>9}{rank:>10}")
    if rows:
        mean_abs = sum(abs(r["d_macro"]) for r in rows) / len(rows)
        print(f"\n[compare] |delta macro| medio = {mean_abs:.4f} · maior = {abs(rows[0]['d_macro']):.4f} "
              f"({rows[0]['config']})")

    if res["contrasts"]:
        print(f"\n{'contraste (A - B)':<48}{'macro ant':>10}{'macro nov':>10}{'miss ant':>10}{'miss nov':>10}"
              f"  {'sinal macro':<13}sinal missense")
        for c in res["contrasts"]:
            print(f"{c['contrast']:<48}{_d(c['macro_old']):>10}{_d(c['macro_new']):>10}"
                  f"{_d(c[PANEL + '_old']):>10}{_d(c[PANEL + '_new']):>10}"
                  f"  {FLAG[c['sinal']['macro']]:<13}{FLAG[c['sinal'][PANEL]]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
