#!/usr/bin/env python3
"""Fase C-1: os dois experimentos que precisam dos VETORES focais. Roda no notebook (so CPU).

Os `probe_focal_*.npz` tem 147 MB e nao precisam viajar -- aqui a algebra e feita onde eles estao
e sai um JSON + um parquet pequenos.

EXPERIMENTO A -- alelo vs alelo (fecha o E2)
    Ate agora comparamos MAGNITUDES escalares por alelo. O "caso ideal" do Eduardo pede a
    distancia entre os VETORES: ref, alt1, alt2 -> as 3 distancias par-a-par > 0. A prova forte e
    ``d(alt1, alt2) > 0``: mostra que o modelo distingue QUAL alelo, nao so "mudou algo".
    Decomposto em h_up (contextual) e h_pure (trivial), porque em h_pure d(alt1,alt2) > 0 e
    garantido por construcao e nao prova nada.

EXPERIMENTO B -- o mesmo sitio com mais contexto (das anotacoes da reuniao do Eduardo)
    "aumentar o contexto e pegar as mesmas posicoes para ver se o vetor naquela posicao e igual...
    para ver se o contexto influenciou a posicao". Sem variante nenhuma: `emb_ref` da MESMA
    posicao em janelas diferentes.

    O problema: a PE e senoidal e o indice focal muda entre janelas centradas (511/1023/2047/
    8191/16383). Como ``h_pure = Linear(token_emb + pos_emb)`` e pointwise, o vetor mudaria por
    causa do INDICE, nao do contexto -- positivo trivial. Por isso o layout `matched` fixa o
    indice em 2047 nas tres janelas. Previsao analitica: no matched, ``h_pure(ref)`` no focal e
    BIT-IDENTICO entre 4k/16k/32k (mesma base, mesmo indice) => toda diferenca restante e
    contextual, pura. O script verifica isso antes de reportar.

    Reportamos os dois layouts: a diferenca entre eles MEDE o artefato de position encoding.

EXPERIMENTO C -- a direcao do Delta e estavel com o contexto?
    Alem da magnitude (ja medida, cai ~4% de 1k a 32k), a direcao: ``cos(Delta_4k, Delta_32k)``.
    Direcao estavel = o modelo tem uma representacao consistente daquela variante, so reescalada.

USO
---
    PYTHONPATH="$WORK" "$PY" scripts/probe_analyze_focal.py \
        --run-dir ~/probe/run --sites ~/probe/run/probe_sites.parquet \
        --out-dir ~/probe/focal_analysis
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CENTERED = (1024, 2048, 4096, 16384, 32768)
MATCHED = (4096, 16384, 32768)
BASES = ("A", "C", "G", "T")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", type=Path, required=True, help="onde estao os probe_focal_*.npz")
    p.add_argument("--sites", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    import numpy as np
    import pandas as pd

    args.out_dir.mkdir(parents=True, exist_ok=True)
    sites = pd.read_parquet(args.sites)
    D_MODEL = 384

    def load(layout: str, bp: int):
        z = np.load(args.run_dir / f"probe_focal_{layout}_{bp}.npz")
        return z["pre"].astype(np.float64), [str(x) for x in z["site"]]

    pre: dict[tuple[str, int], np.ndarray] = {}
    order: list[str] | None = None
    for layout, wins in (("centered", CENTERED), ("matched", MATCHED)):
        for bp in wins:
            arr, sl = load(layout, bp)
            if order is None:
                order = sl
            elif sl != order:
                raise SystemExit(f"ordem de sitios difere em {layout}_{bp}")
            pre[(layout, bp)] = arr
    sites = sites.set_index("site").loc[order].reset_index()
    ref_row = np.array([BASES.index(r) for r in sites["ref"]])
    n = len(sites)
    print(f"[focal] {n:,} sitios x {len(pre)} layouts carregados")

    def cos_dist(a, b):
        """1 - cos, linha a linha. a,b: [n, d]."""
        na, nb = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
        return 1.0 - (a * b).sum(1) / np.where(na * nb > 0, na * nb, np.nan)

    def l2(a, b):
        return np.linalg.norm(a - b, axis=1)

    rows = {"site": sites["site"].to_numpy(), "ref": sites["ref"].to_numpy()}
    report: dict = {"n_sites": int(n)}

    # ---------- A: alelo vs alelo -------------------------------------------------------------
    print("\n" + "=" * 92)
    print("A -- ALELO vs ALELO: as 3 distancias par-a-par do caso ideal do Eduardo")
    print("=" * 92)
    idx = np.arange(n)
    a_rep: dict = {}
    for bp in CENTERED:
        h = pre[("centered", bp)][:, :, :D_MODEL]
        p_ = pre[("centered", bp)][:, :, D_MODEL:]
        alts = np.array([[b for b in range(4) if b != r] for r in ref_row])  # [n, 3]
        d_ref_alt, d_alt_alt, d_alt_alt_pure = [], [], []
        for k in range(3):
            d_ref_alt.append(cos_dist(h[idx, ref_row], h[idx, alts[:, k]]))
        for k, j in ((0, 1), (0, 2), (1, 2)):
            d_alt_alt.append(cos_dist(h[idx, alts[:, k]], h[idx, alts[:, j]]))
            d_alt_alt_pure.append(cos_dist(p_[idx, alts[:, k]], p_[idx, alts[:, j]]))
        ra, aa = np.array(d_ref_alt), np.array(d_alt_alt)
        aap = np.array(d_alt_alt_pure)
        a_rep[bp] = {
            "d_ref_alt_median": float(np.median(ra)), "d_ref_alt_min": float(ra.min()),
            "d_alt_alt_median": float(np.median(aa)), "d_alt_alt_min": float(aa.min()),
            "d_alt_alt_hpure_median": float(np.median(aap)),
            "all_three_positive_frac": float(((ra > 0).all(0) & (aa > 0).all(0)).mean()),
            "alt_alt_over_ref_alt": float(np.median(aa) / np.median(ra)),
        }
        if bp == 4096:
            rows["d_ref_alt_med_4096"] = np.median(ra, axis=0)
            rows["d_alt_alt_med_4096"] = np.median(aa, axis=0)
    print(f"{'janela':>7} {'d(ref,alt) med':>15} {'d(alt1,alt2) med':>17} {'min d(alt,alt)':>15} "
          f"{'3 distintos':>12} {'razao alt-alt/ref-alt':>22}")
    for bp, v in a_rep.items():
        print(f"{bp:>7} {v['d_ref_alt_median']:15.4f} {v['d_alt_alt_median']:17.4f} "
              f"{v['d_alt_alt_min']:15.6f} {v['all_three_positive_frac']:12.1%} "
              f"{v['alt_alt_over_ref_alt']:22.3f}")
    print(f"\n  (em h_pure, d(alt1,alt2) mediano = {a_rep[4096]['d_alt_alt_hpure_median']:.4f} "
          "-- positivo por CONSTRUCAO; o numero que vale e o de h_up acima)")
    report["A_allele_vs_allele"] = a_rep

    # ---------- B: emb_ref entre janelas ------------------------------------------------------
    print("\n" + "=" * 92)
    print("B -- O CONTEXTO MUDA O VETOR DA MESMA POSICAO? (emb_ref, sem variante)")
    print("=" * 92)
    hp_matched = [pre[("matched", bp)][idx, ref_row, D_MODEL:] for bp in MATCHED]
    hp_drift = max(float(np.abs(hp_matched[0] - x).max()) for x in hp_matched[1:])
    ok = hp_drift == 0.0
    print(f"  [pre-check] h_pure(ref) no focal, identico entre as 3 janelas CASADAS? "
          f"max|diff|={hp_drift:.3e} -> {'SIM' if ok else 'NAO'}")
    print("     (previsao analitica: mesma base + mesmo indice => mesmo h_pure. Se falhar, o "
          "layout casado esta errado.)")
    report["B_hpure_matched_drift"] = {"max_abs": hp_drift, "ok": ok}

    b_rep: dict = {}
    for layout, wins in (("centered", CENTERED), ("matched", MATCHED)):
        base_bp = 4096
        h0 = pre[(layout, base_bp)][idx, ref_row, :D_MODEL]
        p0 = pre[(layout, base_bp)][idx, ref_row, D_MODEL:]
        for bp in wins:
            if bp == base_bp:
                continue
            h = pre[(layout, bp)][idx, ref_row, :D_MODEL]
            p_ = pre[(layout, bp)][idx, ref_row, D_MODEL:]
            b_rep[f"{layout}_{base_bp}_vs_{bp}"] = {
                "hup_cosine_median": float(np.median(cos_dist(h0, h))),
                "hup_l2_median": float(np.median(l2(h0, h))),
                "hpure_cosine_median": float(np.median(cos_dist(p0, p_))),
                "frac_changed": float((cos_dist(h0, h) > 1e-9).mean()),
            }
    print(f"\n  {'comparacao':<28} {'h_up cos':>11} {'h_up L2':>10} {'h_pure cos':>12} {'mudou':>8}")
    for k, v in b_rep.items():
        print(f"  {k:<28} {v['hup_cosine_median']:11.5f} {v['hup_l2_median']:10.4f} "
              f"{v['hpure_cosine_median']:12.5f} {v['frac_changed']:8.1%}")
    c4k32 = b_rep["centered_4096_vs_32768"]["hup_cosine_median"]
    m4k32 = b_rep["matched_4096_vs_32768"]["hup_cosine_median"]
    print(f"\n  4k -> 32k, h_up cosine:  centrado {c4k32:.5f}  |  casado {m4k32:.5f}")
    print(f"  o casado isola o CONTEXTO puro (indice focal fixo em 2047, h_pure identico).")
    print(f"  fracao do efeito centrado que e artefato de position encoding: "
          f"{max(0.0, 1 - m4k32 / c4k32):.1%}")
    report["B_ref_across_windows"] = b_rep
    report["B_pe_artifact_fraction"] = float(max(0.0, 1 - m4k32 / c4k32))

    # ---------- C: direcao do Delta -----------------------------------------------------------
    print("\n" + "=" * 92)
    print("C -- A DIRECAO DO DELTA E ESTAVEL COM O CONTEXTO?")
    print("=" * 92)
    c_rep: dict = {}
    for layout, wins in (("centered", CENTERED), ("matched", MATCHED)):
        d0 = None
        for bp in wins:
            h = pre[(layout, bp)][:, :, :D_MODEL]
            # Delta medio sobre as 3 bases nao-ref, por sitio: usamos a 1a alternativa canonica
            alt0 = np.array([[b for b in range(4) if b != r][0] for r in ref_row])
            d = h[idx, alt0] - h[idx, ref_row]
            if d0 is None:
                d0, base_bp = d, bp
                continue
            c_rep[f"{layout}_{base_bp}_vs_{bp}"] = {
                "delta_direction_cosine_distance_median": float(np.median(cos_dist(d0, d))),
                "delta_l2_ratio_median": float(np.median(np.linalg.norm(d, axis=1)
                                                         / np.linalg.norm(d0, axis=1))),
            }
    print(f"  {'comparacao':<28} {'1-cos(Delta,Delta)':>20} {'razao ||Delta||':>17}")
    for k, v in c_rep.items():
        print(f"  {k:<28} {v['delta_direction_cosine_distance_median']:20.5f} "
              f"{v['delta_l2_ratio_median']:17.4f}")
    report["C_delta_direction"] = c_rep

    pd.DataFrame(rows).to_parquet(args.out_dir / "focal_per_site.parquet", index=False)
    (args.out_dir / "focal_analysis.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(f"\n[focal] -> {args.out_dir}/focal_analysis.json  (+ focal_per_site.parquet)")
    print("[focal] os dois arquivos sao pequenos; manda os dois de volta.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
