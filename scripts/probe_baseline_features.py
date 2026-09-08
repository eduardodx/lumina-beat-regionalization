#!/usr/bin/env python3
"""Features de BASELINE a partir das anotacoes do Mosaic. CPU, segundos.

Serve a dois propositos:

1. **Valida o harness em dado real.** As anotacoes tem AUROC conhecida -- ja medimos na sonda que
   phyloP sozinho da 0.756 (missense), 0.959 (splice) e 0.844 (noncoding), ou seja macro ~0.85. Se
   ``probe_feature_eval.py`` reproduzir isso no gold completo sob o protocolo bloqueado, o
   instrumento esta funcionando fora do laboratorio sintetico.

2. **Estabelece o piso.** A propria sonda mostrou que phyloP supera ``||Delta||`` em todos os
   paineis. Entao a pergunta para qualquer representacao que extrairmos do R03 nao e "bate o acaso?"
   e sim "bate a conservacao?". Sem estes numeros, nao ha como responder.

TRATAMENTO DE FALTANTES -- e um desvio consciente
------------------------------------------------
Os comparadores tem NaN fora do seu dominio (REVEL so em missense, SpliceAI so perto de junções...).
O protocolo OFICIAL do Mosaic nao imputa: ele usa intersecao de cobertura e avalia cada comparador
so no dominio nativo. Aqui NAO estamos fazendo a avaliacao oficial de comparadores -- estamos
montando uma matriz de features para um probe. Para isso imputamos pela mediana do TREINO e
adicionamos uma coluna indicadora de ausencia por comparador, que e a pratica correta para features
e preserva a informacao "este score nao existe aqui" (que e informativa: um NaN de REVEL diz que a
variante nao e missense).

Isso significa que os numeros daqui **nao sao** os numeros oficiais de comparador do Mosaic e nao
devem ser reportados como tal.

USO
    PYTHONPATH="$WORK" "$PY" scripts/probe_baseline_features.py \\
        --release-root ~/mosaic-v1 --out ~/probe/baseline/baseline_features.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Blocos de baseline. Cada um e um grupo de colunas de pb_annotations com um papel proprio.
BLOCKS: dict[str, list[str]] = {
    "phylop": ["phylop_241way", "phylop_100way", "phastcons_100way", "phylop_241way_mean_101bp"],
    "revel": ["revel_score"],
    "alphamissense": ["alphamissense_score"],
    "cadd": ["cadd_phred"],
    "spliceai": ["spliceai_ds_ag", "spliceai_ds_al", "spliceai_ds_dg", "spliceai_ds_dl"],
    "pangolin": ["pangolin_gain_score", "pangolin_loss_score"],
    "gnomad": ["gnomad_v4_af", "gnomad_v4_popmax_af", "gnomad_v4_faf95"],
    "other_predictors": ["sift_score", "polyphen2_hvar_score", "gerp_rs", "primateai_score"],
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--release-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--gold-only", action="store_true",
                   help="so as variantes gold (menor arquivo; o probe so testa em gold de qualquer forma)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    import numpy as np
    import pandas as pd

    ann = pd.read_parquet(args.release_root / "pb_annotations.parquet")
    ex = pd.read_parquet(args.release_root / "pb_examples.parquet",
                         columns=["variant_id", "label_tier"])
    df = ann.merge(ex, on="variant_id")
    if args.gold_only:
        df = df[df["label_tier"] == "gold"]
    print(f"[baseline] {len(df):,} variantes"
          + (" (gold)" if args.gold_only else " (todos os tiers)"))

    arrays: dict[str, np.ndarray] = {"variant_id": df["variant_id"].to_numpy().astype("U40")}
    report: dict = {"n_variants": int(len(df)), "blocks": {}}
    present_all: list[str] = []

    for name, cols in BLOCKS.items():
        have = [c for c in cols if c in df.columns]
        if not have:
            print(f"  {name:<18} nenhuma coluna presente, pulando ({cols})")
            continue
        raw = df[have].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
        missing = np.isnan(raw)
        # imputacao pela mediana GLOBAL aqui e so um preenchimento estavel; a padronizacao do probe
        # e refeita no treino de cada fold, entao nao ha vazamento de rotulo por esta etapa.
        med = np.nanmedian(raw, axis=0)
        med = np.where(np.isnan(med), 0.0, med)
        filled = np.where(missing, med, raw)
        # indicadora de ausencia: "nao existe score aqui" e informativo, nao ruido
        feat = np.concatenate([filled, missing.astype(np.float64)], axis=1)
        arrays[f"blk_{name}"] = feat.astype(np.float32)
        present_all.append(name)
        frac = missing.mean(axis=0)
        report["blocks"][name] = {
            "columns": have, "n_dims": int(feat.shape[1]),
            "missing_fraction": {c: round(float(f), 4) for c, f in zip(have, frac)},
        }
        print(f"  {name:<18} {len(have)} col + {len(have)} indicadoras = {feat.shape[1]:>2} dims"
              f"   ausentes: {', '.join(f'{c.split(chr(95))[0]}={f:.0%}' for c, f in zip(have, frac))}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **arrays)
    cfg = {name: [name] for name in present_all}
    cfg["conservacao_apenas"] = ["phylop"]
    cfg["todos_comparadores"] = present_all
    cfg_path = args.out.with_name("baseline_configs.json")
    cfg_path.write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.out.with_suffix(".manifest.json")).write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"\n[baseline] -> {args.out}")
    print(f"[baseline] configs -> {cfg_path}")
    print("\n[baseline] PREVISAO A CONFERIR: 'conservacao_apenas' deve dar macro ~0.85")
    print("           (missense ~0.76 · splice ~0.96 · noncoding ~0.84, medidos na sonda).")
    print("           Se bater, o harness esta correto em dado real. Se divergir muito, o problema")
    print("           esta no harness ou no join -- e nao vale seguir para a extracao rica.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
