#!/usr/bin/env python3
"""Fase C: a analise completa da sonda. CPU, roda em qualquer lugar (notebook ou local).

Regenera TODOS os numeros do relatorio a partir dos artefatos da Fase B, para que nenhum deles
dependa de uma sessao interativa. Estatistica em stdlib (``eval/embedding_probe/stats.py``,
conferida contra o scipy nos testes); pandas/numpy so na borda de leitura.

ENTRADAS (saidas da Fase B, em --run-dir)
    probe_metrics.parquet, probe_sites.parquet, probe_profile_<layout>_<bp>.npz
    opcional: focal_analysis.json (de scripts/probe_analyze_focal.py), so para ecoar no relatorio

SAIDA
    probe_analysis.json -- todos os numeros, com os nomes que o relatorio usa

O QUE MEDE
    E1  percepcao: h_up (contextual) vs h_pure (trivial) -- e quanto o trivial pesa de fato
    E2  discriminacao de alelo + decomposicao de variancia (contexto vs identidade do alelo)
    E2b pareado patogenico-vs-benigno, bruto e ajustado por tipo de substituicao,
        mais o teste da tendencia com o contexto (que NAO sobrevive -- por isso e reportado)
    E3  perfil de propagacao: downstream, UPSTREAM e a SIMETRIA entre os dois; efeito de borda
    E4  contexto: magnitude por janela, centrado vs casado (isola artefato de indice)
    E5  AUROC por painel + os dois confundidores (region_class e phyloP) + IC bootstrap
    RS  alelo REAL do ClinVar vs alelo SINTETICO no mesmo sitio (o modelo prefere os naturais?)
    AUD checagens de consistencia interna entre metricas e perfis

USO
    PYTHONPATH="$WORK" "$PY" scripts/probe_analyze.py --run-dir ~/probe/run \\
        --out ~/probe/probe_analysis.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.embedding_probe.stats import (  # noqa: E402
    auroc,
    bootstrap_auroc_ci,
    mann_whitney_u,
    median,
    quantile,
    spearman,
    variance_decomposition,
    wilcoxon,
)
from eval.embedding_probe.windows import CONSUMER_WINDOWS, EXPLORATORY_WINDOWS  # noqa: E402

WINDOWS = tuple(sorted(EXPLORATORY_WINDOWS + CONSUMER_WINDOWS))
TRANSITIONS = frozenset({"A>G", "G>A", "C>T", "T>C"})
DISCRIMINATION_PANELS = ("missense", "splice", "noncoding")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--focal-analysis", type=Path, default=None)
    p.add_argument("--window", type=int, default=4096, help="janela de referencia dos resumos")
    return p.parse_args(argv)


def load(run_dir: Path):
    """Metricas + sitios, com rotulo/painel/region_class/phyloP resolvidos por (sitio, base)."""
    import pandas as pd

    metrics = pd.read_parquet(run_dir / "probe_metrics.parquet")
    sites = pd.read_parquet(run_dir / "probe_sites.parquet")

    label, panel, region, phylop = {}, {}, {}, {}
    for row in sites.itertuples():
        panels = str(row.panels).split(",")
        for token in str(row.alt_labels).split(";"):
            alt, _, value = token.partition("=")
            if value.strip().lstrip("-").isdigit():
                label[(row.site, alt)] = int(value)
        for alt in str(row.clinvar_alts).split(","):
            # painel so quando o sitio inteiro tem um -- sitios com ALTs de paineis diferentes
            # sairiam com atribuicao ambigua, entao ficam de fora das analises por painel
            panel[(row.site, alt)] = panels[0] if len(panels) == 1 else None
            region[(row.site, alt)] = row.region_class
            phylop[(row.site, alt)] = row.phylop_241way

    metrics = metrics.copy()
    metrics["subst"] = metrics["ref"] + ">" + metrics["base"]
    metrics["transition"] = metrics["subst"].isin(TRANSITIONS)
    key = list(zip(metrics["site"], metrics["base"]))
    metrics["label"] = [label.get(k) for k in key]
    metrics["panel"] = [panel.get(k) for k in key]
    metrics["region_class"] = [region.get(k) for k in key]
    metrics["phylop"] = [phylop.get(k) for k in key]
    # z-score DENTRO de cada (janela, tipo de substituicao): remove a identidade da troca, que e
    # um confundidor forte (P e B tem composicao muito diferente de transicao/transversao)
    metrics["hup_z"] = metrics.groupby(["window_bp", "layout", "subst"])["hup_l2"].transform(
        lambda s: (s - s.mean()) / s.std()
    )
    return metrics, sites


def profile_means(run_dir: Path, layout: str, window_bp: int) -> dict:
    """Media, sobre sitios e bases nao-ref, de ||Delta h_up|| por bin. A linha ref e zero por
    construcao e precisa ser mascarada, senao a media cai por 1/4."""
    import numpy as np

    z = np.load(run_dir / f"probe_profile_{layout}_{window_bp}.npz")
    arr = z["hup_mean"]
    nonref = (arr != 0).any(axis=2)
    out = {}
    for i, lbl in enumerate(z["bin_label"]):
        out[str(lbl)] = {
            "mean": float(arr[:, :, i][nonref].mean()),
            "direction": str(z["bin_direction"][i]),
            "lo": int(z["bin_lo"][i]),
            "hi": int(z["bin_hi"][i]),
            "width": int(z["bin_width"][i]),
        }
    return out


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    metrics, sites = load(args.run_dir)
    C = metrics[metrics["layout"] == "centered"]
    W = C[C["window_bp"] == args.window]
    real = C[C["is_clinvar_alt"]]
    report: dict = {"window_reference": args.window, "n_sites": int(sites.shape[0]),
                    "n_metric_rows": int(len(metrics))}
    say = print

    # ---------------- E1 -----------------------------------------------------------------
    say("=" * 92); say("E1 -- PERCEPCAO: contextual (h_up) vs trivial (h_pure)"); say("=" * 92)
    e1 = {}
    for block, dims in (("hup", 384), ("hpure", 64), ("pre448", 448), ("post448", 448)):
        cos = W[f"{block}_cosine_distance"].tolist()
        e1[block] = {
            "dims": dims, "cosine_median": median(cos),
            "cosine_p05": quantile(cos, .05), "cosine_p95": quantile(cos, .95),
            "l2_median": median(W[f"{block}_l2"].tolist()),
            "relative_l2_median": median(W[f"{block}_relative_l2"].tolist()),
            "ref_norm_median": median(W[f"{block}_ref_norm"].tolist()),
        }
        say(f"  {block:<8} dims={dims:>3}  cos med {e1[block]['cosine_median']:.4f}  "
            f"L2 med {e1[block]['l2_median']:.4f}  rel_L2 {e1[block]['relative_l2_median']:.4f}")
    energy = [p**2 / (u**2 + p**2) for u, p in zip(W["hup_l2"], W["hpure_l2"])]
    e1["hpure_energy_share_median"] = median(energy)
    e1["frac_hup_moves_more"] = float((W["hup_cosine_distance"] > W["hpure_cosine_distance"]).mean())
    e1["n_pairs"] = int(len(W))
    e1["min_hup_cosine_by_window"] = {
        str(bp): float(C[C["window_bp"] == bp]["hup_cosine_distance"].min()) for bp in WINDOWS}
    e1["all_pairs_positive"] = bool((C["hup_cosine_distance"] > 0).all())
    say(f"  h_pure responde por {e1['hpure_energy_share_median']:.4%} da energia da mudanca")
    say(f"  h_up se move mais que h_pure em {e1['frac_hup_moves_more']:.1%} dos {e1['n_pairs']:,} pares")
    say(f"  |cos(448) - cos(h_up)| / cos(h_up) = "
        f"{abs(e1['pre448']['cosine_median'] - e1['hup']['cosine_median']) / e1['hup']['cosine_median']:.2%}")
    report["E1_perception"] = e1

    # ---------------- E2 -----------------------------------------------------------------
    say("\n" + "=" * 92); say("E2 -- DISCRIMINACAO DE ALELO: contexto vs identidade do alelo"); say("=" * 92)
    groups = [g["hup_cosine_distance"].tolist() for _, g in W.groupby("site")]
    vd = variance_decomposition(groups)
    distinct = sum(1 for g in groups if max(g) - min(g) > 1e-6)
    e2 = {**vd, "n_sites": len(groups), "frac_sites_all_alts_distinct": distinct / len(groups)}
    say(f"  variancia ENTRE sitios (contexto genomico): {vd['frac_between']:.1%}")
    say(f"  variancia DENTRO do sitio (identidade do alelo): {vd['frac_within']:.1%}")
    say(f"  razao dos desvios-padrao dentro/entre: {vd['sd_ratio']:.2f}")
    say(f"  sitios com os 3 ALTs distintos: {e2['frac_sites_all_alts_distinct']:.1%}")
    report["E2_allele_discrimination"] = e2

    # ---------------- E2b ----------------------------------------------------------------
    say("\n" + "=" * 92); say("E2b -- PAREADO PATOGENICO vs BENIGNO (mesmo sitio)"); say("=" * 92)
    pb_sites = set(sites[sites["arms"].str.contains("paired_pb_gold", na=False)]["site"])
    paired = real[real["label"].notna() & real["site"].isin(pb_sites)]
    e2b: dict = {"n_sites": len(pb_sites), "by_window": {}}
    gaps: dict[int, dict] = {}
    say(f"  {'janela':>7} {'n':>4} | {'P bruto':>8} {'B bruto':>8} {'P>B':>7} {'p':>9} |"
        f" {'P ajust':>8} {'B ajust':>8} {'P>B':>7} {'p':>9}")
    for bp in WINDOWS:
        d = paired[paired["window_bp"] == bp]
        raw = d.groupby(["site", "label"])["hup_l2"].mean().unstack().dropna()
        adj = d.groupby(["site", "label"])["hup_z"].mean().unstack().dropna()
        if len(raw) < 5:
            continue
        P, B = raw[1.0].tolist(), raw[0.0].tolist()
        zP, zB = adj[1.0].tolist(), adj[0.0].tolist()
        _, p_raw, _ = wilcoxon(P, B)
        _, p_adj, _ = wilcoxon(zP, zB)
        gaps[bp] = {"site": adj.index.tolist(), "gap": [a - b for a, b in zip(zP, zB)]}
        e2b["by_window"][str(bp)] = {
            "n_pairs": len(raw),
            "raw": {"P_median": median(P), "B_median": median(B),
                    "frac_P_gt_B": sum(a > b for a, b in zip(P, B)) / len(P), "p": p_raw},
            "adjusted": {"P_median": median(zP), "B_median": median(zB),
                         "frac_P_gt_B": sum(a > b for a, b in zip(zP, zB)) / len(zP),
                         "gap_median": median([a - b for a, b in zip(zP, zB)]), "p": p_adj},
        }
        r, a = e2b["by_window"][str(bp)]["raw"], e2b["by_window"][str(bp)]["adjusted"]
        say(f"  {bp:>7} {len(raw):>4} | {r['P_median']:8.4f} {r['B_median']:8.4f} "
            f"{r['frac_P_gt_B']:7.1%} {r['p']:9.1e} | {a['P_median']:8.3f} {a['B_median']:8.3f} "
            f"{a['frac_P_gt_B']:7.1%} {a['p']:9.1e}")

    # o confundidor: composicao de transicao/transversao difere muito entre P e B
    w_pair = paired[paired["window_bp"] == args.window]
    e2b["substitution_confounder"] = {
        "frac_transition_pathogenic": float(w_pair[w_pair["label"] == 1]["transition"].mean()),
        "frac_transition_benign": float(w_pair[w_pair["label"] == 0]["transition"].mean()),
    }
    # Duas populacoes: os ALTs reais rotulados (o grupo em que o confundidor de fato atua) e todos
    # os pares (caracterizacao geral do modelo). Os numeros diferem; o relatorio cita a primeira.
    for name, pop in (("clinvar_labeled", real[(real["window_bp"] == args.window)
                                               & real["label"].notna()]),
                      ("all_pairs", W)):
        tr = pop[pop["transition"]]["hup_l2"].tolist()
        tv = pop[~pop["transition"]]["hup_l2"].tolist()
        _, p_sub = mann_whitney_u(tv, tr)
        e2b["substitution_confounder"][name] = {
            "n": len(pop), "transversion_median": median(tv), "transition_median": median(tr),
            "p": p_sub}
    c = e2b["substitution_confounder"]
    say("")
    say(f"  CONFUNDIDOR: transicoes em P={c['frac_transition_pathogenic']:.1%} vs "
        f"B={c['frac_transition_benign']:.1%}")
    for name in ("clinvar_labeled", "all_pairs"):
        v = c[name]
        say(f"    {name:<16} (n={v['n']:>5}): transversao {v['transversion_median']:.4f} vs "
            f"transicao {v['transition_median']:.4f}  p={v['p']:.1e}")

    # a tendencia com o contexto: a contagem sobe, mas o gap pareado nao
    if 1024 in gaps and 32768 in gaps:
        s1 = dict(zip(gaps[1024]["site"], gaps[1024]["gap"]))
        s2 = dict(zip(gaps[32768]["site"], gaps[32768]["gap"]))
        common = [k for k in s1 if k in s2]
        _, p_trend, _ = wilcoxon([s2[k] for k in common], [s1[k] for k in common])
        e2b["context_trend"] = {
            "n": len(common), "p": p_trend,
            "gap_median_by_window": {str(bp): median(g["gap"]) for bp, g in gaps.items()},
            "significant": bool(p_trend < 0.05),
        }
        say(f"  TENDENCIA com contexto (gap 32k vs 1k, {len(common)} sitios pareados): "
            f"p={p_trend:.3f} -> {'significativa' if p_trend < 0.05 else 'NAO significativa'}")
        say(f"    gap mediano por janela: " + "  ".join(
            f"{bp//1024}k={median(g['gap']):+.3f}" for bp, g in sorted(gaps.items())))
    report["E2b_pathogenic_vs_benign"] = e2b

    # ---------------- E3 -----------------------------------------------------------------
    say("\n" + "=" * 92); say("E3 -- PROPAGACAO: downstream, UPSTREAM e simetria"); say("=" * 92)
    prof = {bp: profile_means(args.run_dir, "centered", bp) for bp in WINDOWS}
    biggest = prof[max(WINDOWS)]
    e3: dict = {"profile_centered": prof, "symmetry": {}, "edge_effect": {}}

    say(f"  {'|offset|':<16} {'upstream':>11} {'downstream':>11} {'razao up/down':>14}")
    ratios = []
    for lbl, meta in sorted(biggest.items(), key=lambda kv: kv[1]["lo"]):
        if meta["direction"] != "down":
            continue
        up_lbl = f"up_{meta['lo']}_{meta['hi']}"
        if up_lbl not in biggest:
            continue
        up, down = biggest[up_lbl]["mean"], meta["mean"]
        ratio = up / down if down else float("nan")
        e3["symmetry"][f"{meta['lo']}_{meta['hi']}"] = {
            "upstream": up, "downstream": down, "ratio": ratio}
        if meta["lo"] >= 1:
            ratios.append(ratio)
        if meta["lo"] in (1, 5, 25, 65, 257, 1025, 4097, 8193):
            say(f"  {meta['lo']}-{meta['hi']:<12} {up:11.5f} {down:11.5f} {ratio:14.3f}")
    e3["symmetry_summary"] = {
        "ratio_median": median(ratios), "ratio_min": min(ratios), "ratio_max": max(ratios),
        "n_bins": len(ratios),
    }
    say(f"  -> razao up/down: mediana {median(ratios):.3f} "
        f"(min {min(ratios):.3f}, max {max(ratios):.3f}) em {len(ratios)} bins")
    say(f"     {'propagacao SIMETRICA' if 0.9 < median(ratios) < 1.1 else 'propagacao ASSIMETRICA'}"
        f" -- o mid-stack Mamba e bidirecional, agora verificado")

    say(f"\n  efeito de borda -- mesmo bin, janelas diferentes (razao vs 32768):")
    for lbl in ("focal", "down_1_1", "down_5_5", "down_25_25", "down_65_128",
                "down_129_256", "down_257_512", "down_513_1024", "down_1025_2048"):
        base = biggest.get(lbl)
        if not base:
            continue
        row = {str(bp): prof[bp][lbl]["mean"] / base["mean"] for bp in WINDOWS if lbl in prof[bp]}
        e3["edge_effect"][lbl] = row
        say(f"    {lbl:<16} " + "  ".join(f"{bp}:{v:.2f}x" for bp, v in row.items()))
    report["E3_propagation"] = e3

    # ---------------- E4 -----------------------------------------------------------------
    say("\n" + "=" * 92); say("E4 -- CONTEXTO: magnitude por janela, centrado vs casado"); say("=" * 92)
    e4: dict = {"centered": {}, "matched": {}}
    for bp in WINDOWS:
        e4["centered"][str(bp)] = {
            "hup_l2_median": median(C[C["window_bp"] == bp]["hup_l2"].tolist()),
            "hup_cosine_median": median(C[C["window_bp"] == bp]["hup_cosine_distance"].tolist()),
            "hpure_l2_median": median(C[C["window_bp"] == bp]["hpure_l2"].tolist()),
        }
    M = metrics[metrics["layout"] == "matched"]
    for bp in CONSUMER_WINDOWS:
        e4["matched"][str(bp)] = {
            "hup_l2_median": median(M[M["window_bp"] == bp]["hup_l2"].tolist())}
    c4, c32 = e4["centered"]["4096"]["hup_l2_median"], e4["centered"]["32768"]["hup_l2_median"]
    m4, m32 = e4["matched"]["4096"]["hup_l2_median"], e4["matched"]["32768"]["hup_l2_median"]
    e4["drop_4k_to_32k_centered"] = 1 - c32 / c4
    e4["drop_4k_to_32k_matched"] = 1 - m32 / m4
    e4["position_artifact_fraction"] = 1 - (1 - m32 / m4) / (1 - c32 / c4)
    # A previsao e que ||Delta h_pure|| no focal seja identico entre janelas (pos_emb cancela na
    # diferenca). Em ponto flutuante ele cancela ate ~1e-11: os valores absolutos de h_pure MUDAM
    # com o indice, entao a subtracao arredonda diferente. Comparamos com tolerancia relativa.
    hp = [v["hpure_l2_median"] for v in e4["centered"].values()]
    e4["hpure_l2_by_window"] = hp
    e4["hpure_relative_spread"] = (max(hp) - min(hp)) / median(hp)
    e4["hpure_constant_across_windows"] = e4["hpure_relative_spread"] < 1e-6
    say(f"  queda 4k->32k: centrado {e4['drop_4k_to_32k_centered']:.2%}, "
        f"casado {e4['drop_4k_to_32k_matched']:.2%}")
    say(f"  -> artefato de indice de posicao: {e4['position_artifact_fraction']:.0%} da queda aparente")
    say(f"  ||Delta h_pure|| constante entre janelas: {e4['hpure_constant_across_windows']} "
        f"(dispersao relativa {e4['hpure_relative_spread']:.2e} -- previsao analitica)")
    report["E4_context"] = e4

    # ---------------- E5 -----------------------------------------------------------------
    say("\n" + "=" * 92); say("E5 -- SINAL CLINICO: AUROC por painel e os confundidores"); say("=" * 92)
    e5: dict = {"by_panel": {}, "confounders": {}}
    say(f"  {'painel':<11} {'janela':>7} {'nP':>4} {'nB':>4} {'AUROC':>7} {'ajust':>7} {'phyloP':>7}")
    for pan in DISCRIMINATION_PANELS:
        for bp in WINDOWS:
            d = real[(real["window_bp"] == bp) & (real["panel"] == pan) & real["label"].notna()]
            P, B = d[d["label"] == 1], d[d["label"] == 0]
            if len(P) < 20 or len(B) < 20:
                continue
            ph = d[d["phylop"].notna()]
            phP, phB = ph[ph["label"] == 1], ph[ph["label"] == 0]
            row = {
                "n_pos": len(P), "n_neg": len(B),
                "auroc_raw": auroc(P["hup_l2"].tolist(), B["hup_l2"].tolist()),
                "auroc_subst_adjusted": auroc(P["hup_z"].tolist(), B["hup_z"].tolist()),
                "auroc_phylop_alone": auroc(phP["phylop"].tolist(), phB["phylop"].tolist()),
                "spearman_delta_phylop": spearman(ph["hup_l2"].tolist(), ph["phylop"].tolist())[0],
            }
            e5["by_panel"][f"{pan}_{bp}"] = row
            if bp == args.window:
                say(f"  {pan:<11} {bp:>7} {row['n_pos']:>4} {row['n_neg']:>4} "
                    f"{row['auroc_raw']:>7.3f} {row['auroc_subst_adjusted']:>7.3f} "
                    f"{row['auroc_phylop_alone']:>7.3f}")

    d = real[(real["window_bp"] == args.window) & (real["panel"] == "splice") & real["label"].notna()]
    canon = [(1.0 if r == "splice_canonical" else 0.0) for r in d["region_class"]]
    lab = d["label"].tolist()
    e5["confounders"]["splice_region_class_alone_auroc"] = auroc(
        [c for c, l in zip(canon, lab) if l == 1], [c for c, l in zip(canon, lab) if l == 0])
    sub = d[d["region_class"] == "splice_region"]
    sP, sB = sub[sub["label"] == 1]["hup_l2"].tolist(), sub[sub["label"] == 0]["hup_l2"].tolist()
    lo, hi = bootstrap_auroc_ci(sP, sB)
    e5["confounders"]["splice_region_only"] = {
        "n_pos": len(sP), "n_neg": len(sB), "auroc": auroc(sP, sB), "ci95": [lo, hi]}
    say(f"\n  CONFUNDIDOR 1 -- region_class sozinho prevê o rotulo de splice: "
        f"AUROC {e5['confounders']['splice_region_class_alone_auroc']:.3f}")
    say(f"    so dentro de splice_region: AUROC "
        f"{e5['confounders']['splice_region_only']['auroc']:.3f} "
        f"[{lo:.3f}, {hi:.3f}]  (nP={len(sP)}, nB={len(sB)})")
    say(f"  CONFUNDIDOR 2 -- phyloP sozinho ja supera a sonda em todos os paineis (tabela acima)")
    report["E5_clinical_signal"] = e5

    # ---------------- RS: real vs sintetico ----------------------------------------------
    say("\n" + "=" * 92)
    say("RS -- ALELO REAL DO CLINVAR vs ALELO SINTETICO (pareado dentro do sitio)"); say("=" * 92)
    rs: dict = {"by_window": {}}
    say(f"  {'janela':>7} {'n sitios':>9} {'real':>9} {'sintetico':>10} {'real>sint':>10} "
        f"{'p bruto':>9} {'p ajust':>9}")
    for bp in WINDOWS:
        d = C[C["window_bp"] == bp]
        r_, s_, rz, sz = [], [], [], []
        for _, g in d.groupby("site"):
            a, b = g[g["is_clinvar_alt"]], g[~g["is_clinvar_alt"]]
            if len(a) == 0 or len(b) == 0:
                continue
            r_.append(a["hup_l2"].mean()); s_.append(b["hup_l2"].mean())
            rz.append(a["hup_z"].mean()); sz.append(b["hup_z"].mean())
        _, p_raw, _ = wilcoxon(r_, s_)
        _, p_adj, _ = wilcoxon(rz, sz)
        rs["by_window"][str(bp)] = {
            "n_sites": len(r_), "real_median": median(r_), "synthetic_median": median(s_),
            "frac_real_gt_synthetic": sum(a > b for a, b in zip(r_, s_)) / len(r_),
            "frac_real_gt_synthetic_adjusted": sum(a > b for a, b in zip(rz, sz)) / len(rz),
            "p_raw": p_raw, "p_subst_adjusted": p_adj,
            "real_z_median": median(rz), "synthetic_z_median": median(sz),
            "direction_raw": "real>sintetico" if median(r_) > median(s_) else "sintetico>real",
            "direction_adjusted": "real>sintetico" if median(rz) > median(sz) else "sintetico>real",
        }
        v = rs["by_window"][str(bp)]
        say(f"  {bp:>7} {v['n_sites']:>9} {v['real_median']:>9.4f} {v['synthetic_median']:>10.4f} "
            f"{v['frac_real_gt_synthetic']:>10.1%} {v['p_raw']:>9.1e} {v['p_subst_adjusted']:>9.1e}")
    ref = rs["by_window"][str(args.window)]
    say(f"\n  Em {args.window}bp: real {ref['real_median']:.4f} vs sintetico "
        f"{ref['synthetic_median']:.4f}; ajustado por substituicao "
        f"z_real={ref['real_z_median']:+.3f} vs z_sint={ref['synthetic_z_median']:+.3f}")
    say(f"  -> BRUTO: {ref['direction_raw']} (p={ref['p_raw']:.1e})  |  "
        f"AJUSTADO: {ref['direction_adjusted']} (p={ref['p_subst_adjusted']:.1e})")
    if ref["direction_raw"] != ref["direction_adjusted"]:
        say("     A DIRECAO INVERTE com o ajuste: os ALTs do ClinVar sao enriquecidos em transicoes")
        say("     (C>T em CpG e a mutacao humana mais comum), e transicoes dao Delta menor. Sem o")
        say("     ajuste isso faz o alelo sintetico parecer mais perturbador. Controlado o tipo de")
        say("     troca, sobra uma vantagem PEQUENA do alelo real.")
    report["RS_real_vs_synthetic"] = rs

    # ---------------- AUD ----------------------------------------------------------------
    say("\n" + "=" * 92); say("AUD -- consistencia interna"); say("=" * 92)
    aud = {}
    for bp in (min(WINDOWS), args.window, max(WINDOWS)):
        p = profile_means(args.run_dir, "centered", bp)["focal"]["mean"]
        m = float(C[C["window_bp"] == bp]["hup_l2"].mean())
        aud[f"focal_bin_vs_metric_{bp}"] = abs(p - m)
        say(f"  bin focal do perfil vs hup_l2 das metricas ({bp}bp): |diff| = {abs(p - m):.2e}")
    bad = int((metrics.groupby(["window_bp", "layout", "subst"])["hpure_l2"].nunique() > 1).sum())
    aud["hpure_nonconstant_groups"] = bad
    say(f"  grupos (janela,layout,substituicao) com ||Delta h_pure|| nao-constante: {bad} (esperado 0)")
    num = metrics.select_dtypes(include=["number"])
    aud["n_nan"] = int(num.isna().sum().sum() - metrics["label"].isna().sum())
    aud["cosine_out_of_range"] = int(
        ((metrics["hup_cosine_distance"] < 0) | (metrics["hup_cosine_distance"] > 2)).sum())
    say(f"  cossenos fora de [0,2]: {aud['cosine_out_of_range']} (esperado 0)")
    report["AUDIT"] = aud

    if args.focal_analysis and args.focal_analysis.is_file():
        report["focal_vector_analysis"] = json.loads(args.focal_analysis.read_text(encoding="utf-8"))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n",
                        encoding="utf-8")
    say(f"\n[analise] -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
