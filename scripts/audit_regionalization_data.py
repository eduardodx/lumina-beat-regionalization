#!/usr/bin/env python3
"""Auditoria READ-ONLY dos dados da regionalizacao R03, antes do redesenho sem adapter ClinVar e sem fusion.

Roda no notebook (pandas + pyarrow; sem GPU, sem S3). So escreve o JSON de saida.

POR QUE EXISTE
--------------
O redesenho (tirar o adapter ClinVar e a fusion, regionalizar com gnomAD e ABraOM) passa a depender
das colunas de frequencia e do T_nonBR congelado. Lendo o codigo, cinco fatos que o desenho precisa
ficaram LIDOS mas nao MEDIDOS. Este script mede, sem supor:

  [A] af_gnomad condicional ao ABraOM. Em prepare_regional_clinvar_dataset.py o af_gnomad chega SO pelo
      merge (how="inner") com o indice ABraOM v2, e depois vai por left-join ao ClinVar. Se isso vale no
      dado, o "91% do br_only ausente do gnomAD" da Fase 0 quer dizer "ausente do indice ABraOM": o
      gnomAD nunca foi consultado para essas variantes.
  [B] Pareamento indireto por ABraOM no T_nonBR. O matcher usa log10(af_gnomad.fillna(0)) na distancia.
      Se [A] vale, essa feature carrega a presenca no ABraOM e os pares teriam casado por ela -- o que o
      paragrafo 6.7 do Eduardo proibe ("nao parear por presenca no ABraOM"). Compara a concordancia de
      abraom_present nos pares com a esperada se o matcher sorteasse dentro do estrato exato
      (gene, label, tipo).
  [C] Sobreposicao com o gold do Mosaic. A receita de extracao de embeddings foi escolhida no gold do
      Mosaic; se ele contem variantes do T_BR/T_nonBR, a escolha tocou o teste da regionalizacao.
  [D] Composicao por tipo de variante. A receita de 172 dims foi validada SO em SNV.
  [E] (opcional, --abraom-tsv) Piso de AF do ABraOM: AF minima do TSV cru SABE-WGS-1171, e quantas
      variantes do T_BR estao no TSV cru mas marcadas abraom_present=False pelo indice v2.

O QUE ESTE SCRIPT NAO PROVA
---------------------------
[B] mede EXCESSO de concordancia sobre um sorteio dentro do estrato. O matcher real tambem usou
submitters, estrelas e consequencia, que podem correlacionar com a presenca no ABraOM por motivos
legitimos. Excesso aqui e evidencia do mecanismo, nao prova; o teste decisivo e re-parear sem af_gnomad
e comparar os dois T_nonBR.

USO (notebook)
--------------
    PYTHONPATH="$WORK" "$PY" scripts/audit_regionalization_data.py \
        --br <br_only que o matcher usou> --nonbr <nonbr_only que o matcher usou> \
        --pairs ~/artifacts/fase0/t_nonbr_matched_soft.parquet \
        --splits ~/artifacts/fase0/clinvar_splits/clinvar_splits_combined.parquet \
        --mosaic-examples ~/mosaic-v1/pb_examples.parquet \
        --out ~/artifacts/redesenho/audit_dados.json

Passe em --nonbr o MESMO arquivo que o matcher usou: o baseline de [B] e calculado nesse pool.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

AF_THRESHOLDS = (0.0005, 0.001, 0.005, 0.01)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--br", type=Path, required=True, help="slice br_only (a mesma que o matcher usou)")
    p.add_argument("--nonbr", type=Path, required=True, help="slice nonbr_only (a MESMA que o matcher usou)")
    p.add_argument("--pairs", type=Path, required=True, help="T_nonBR congelado (t_nonbr_matched_soft)")
    p.add_argument("--splits", type=Path, default=None, help="clinvar_splits_combined.parquet")
    p.add_argument("--mosaic-examples", type=Path, default=None, help="pb_examples.parquet do release Mosaic")
    p.add_argument("--abraom-tsv", type=Path, default=None, help="SABE1171.Abraom.clean.tsv (opcional)")
    p.add_argument("--exclude-chrom", default="8")
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args(argv)


# ------------------------------------------------------------------------------------------------
# normalizacao
# ------------------------------------------------------------------------------------------------


def as_bool(series):
    """abraom_present chega como bool, boolean ou texto conforme o parquet. NA conta como False."""
    if series.dtype == bool:
        return series
    if str(series.dtype) == "boolean":
        return series.fillna(False).astype(bool)
    text = series.astype("string").str.strip().str.lower()
    return text.isin(["true", "1", "1.0", "t", "yes"]).astype(bool)


def norm_chrom(series):
    """Mesma convencao do variant_key regional: sem prefixo chr, X/Y/M maiusculos, numero como texto."""
    s = series.astype("string").str.strip().str.lower().str.removeprefix("chr")
    s = s.replace({"mt": "m", "23": "x", "24": "y"})
    return s.where(~s.isin(["x", "y", "m"]), s.str.upper())


def key_chrom(keys):
    """Cromossomo tirado do proprio variant_key (chrom:pos:ref:alt), em minusculas e sem prefixo chr."""
    return keys.astype("string").str.split(":").str[0].str.lower().str.removeprefix("chr")


def snv_from_key(keys):
    parts = keys.astype("string").str.split(":")
    return (parts.str[2].str.len() == 1) & (parts.str[3].str.len() == 1)


def build_key(chrom, pos, ref, alt):
    return (norm_chrom(chrom) + ":" + pos.astype("Int64").astype("string") + ":"
            + ref.astype("string").str.strip().str.upper() + ":"
            + alt.astype("string").str.strip().str.upper())


def load_slice(path: Path, exclude_chrom: str):
    import pandas as pd

    df = pd.read_parquet(path)
    if "variant_key" not in df.columns:
        raise SystemExit(f"{path}: sem variant_key; colunas: {sorted(df.columns)}")
    n_raw = len(df)
    df = df.drop_duplicates("variant_key").reset_index(drop=True)   # unidade = variante canonica (5.2)
    main = df[key_chrom(df["variant_key"]) != exclude_chrom].reset_index(drop=True)
    return main, {"linhas": n_raw, "variantes_unicas": len(df), "sem_chr_excluido": len(main)}


# ------------------------------------------------------------------------------------------------
# [A] af_gnomad condicional ao ABraOM
# ------------------------------------------------------------------------------------------------


def audit_af_gnomad_conditional(df, name: str) -> dict:
    import pandas as pd

    out: dict = {"slice": name, "n": int(len(df))}
    missing = [c for c in ("af_gnomad", "abraom_present") if c not in df.columns]
    if missing:
        out["erro"] = f"faltam {missing}"
        return out
    af = pd.to_numeric(df["af_gnomad"], errors="coerce")
    present = as_bool(df["abraom_present"])
    filled = af.notna()
    n_absent = int((~present).sum())
    out.update({
        "abraom_present": int(present.sum()),
        "af_gnomad_preenchido": int(filled.sum()),
        "af_gnomad_zero_ou_na": int((af.fillna(0.0) <= 0).sum()),
        "tabela": {
            "abraom_presente__af_preenchido": int((present & filled).sum()),
            "abraom_presente__af_na": int((present & ~filled).sum()),
            "abraom_ausente__af_preenchido": int((~present & filled).sum()),
            "abraom_ausente__af_na": int((~present & ~filled).sum()),
        },
        "p_af_preenchido_se_abraom_ausente": (float((~present & filled).sum()) / n_absent) if n_absent else None,
    })
    p = out["p_af_preenchido_se_abraom_ausente"]
    if p is None:
        out["veredito"] = "sem variantes ausentes do ABraOM para testar"
    elif p <= 0.001:
        out["veredito"] = "CONFIRMADO: af_gnomad so existe onde ha ABraOM (gnomAD nao foi consultado fora do indice)"
    else:
        out["veredito"] = "NAO confirmado: af_gnomad existe fora do indice ABraOM"
    return out


# ------------------------------------------------------------------------------------------------
# [B] pareamento indireto por ABraOM
# ------------------------------------------------------------------------------------------------


def audit_pair_abraom_concordance(pairs, pool) -> dict:
    import numpy as np
    import pandas as pd

    need_pairs = ["br_abraom_present", "nonbr_abraom_present", "br_GeneSymbol", "br_label", "br_variant_type"]
    need_pool = ["GeneSymbol", "label", "variant_type", "abraom_present"]
    miss_p = [c for c in need_pairs if c not in pairs.columns]
    miss_q = [c for c in need_pool if c not in pool.columns]
    if miss_p or miss_q:
        return {"erro": f"pares sem {miss_p}; pool sem {miss_q}"}

    br_p = as_bool(pairs["br_abraom_present"]).to_numpy()
    nb_p = as_bool(pairs["nonbr_abraom_present"]).to_numpy()
    n = len(pairs)
    p_br, p_nb = float(br_p.mean()), float(nb_p.mean())

    # P(abraom_present | estrato exato) no pool -- o que um sorteio dentro do estrato daria
    strat = pool.assign(
        _g=pool["GeneSymbol"].astype("string"),
        _l=pd.to_numeric(pool["label"], errors="coerce"),
        _t=pool["variant_type"].astype("string"),
        _present=as_bool(pool["abraom_present"]).astype(float),
    ).dropna(subset=["_g", "_l", "_t"])
    q_by = strat.groupby(["_g", "_l", "_t"])["_present"].mean().to_dict()
    keys = zip(pairs["br_GeneSymbol"].astype("string"),
               pd.to_numeric(pairs["br_label"], errors="coerce"),
               pairs["br_variant_type"].astype("string"))
    q = np.array([q_by.get((g, lab, t), np.nan) for g, lab, t in keys], dtype=float)

    def _block(mask) -> dict:
        m = int(mask.sum())
        if m == 0:
            return {"n": 0}
        obs = float((br_p[mask] == nb_p[mask]).mean())
        exp = float(np.where(br_p[mask], q[mask], 1.0 - q[mask]).mean())
        se = math.sqrt(max(exp * (1.0 - exp), 1e-12) / m)
        return {"n": m, "concordancia_observada": obs, "concordancia_esperada_sorteio_no_estrato": exp,
                "excesso": obs - exp, "z": (obs - exp) / se}

    has_q = ~np.isnan(q)
    # So em estratos MISTOS (0<q<1) o matcher tem escolha; em estratos puros a concordancia e forcada.
    mixed = has_q & (np.nan_to_num(q, nan=-1.0) > 0.0) & (np.nan_to_num(q, nan=2.0) < 1.0)
    out = {
        "n_pares": n,
        "p_abraom_present_br": p_br,
        "p_abraom_present_nonbr_pareado": p_nb,
        "concordancia_se_independente": p_br * p_nb + (1 - p_br) * (1 - p_nb),
        "tabela_2x2": {
            "br_sim__nonbr_sim": int((br_p & nb_p).sum()), "br_sim__nonbr_nao": int((br_p & ~nb_p).sum()),
            "br_nao__nonbr_sim": int((~br_p & nb_p).sum()), "br_nao__nonbr_nao": int((~br_p & ~nb_p).sum()),
        },
        "todos_os_estratos": _block(has_q),
        "estratos_mistos": _block(mixed),
        "pares_sem_estrato_no_pool": int((~has_q).sum()),
    }
    blk = out["estratos_mistos"]
    if blk.get("n", 0) == 0:
        out["veredito"] = "sem estratos mistos: nao ha como o matcher ter escolhido por ABraOM"
    elif blk["z"] > 3 and blk["excesso"] > 0.05:
        out["veredito"] = ("EVIDENCIA FORTE de pareamento pela presenca no ABraOM (viola o 6.7): "
                           "re-parear sem af_gnomad e comparar")
    elif blk["z"] > 3:
        out["veredito"] = "excesso detectavel mas pequeno (<0.05): registrar, re-parear e comparar"
    else:
        out["veredito"] = "sem evidencia de pareamento pela presenca no ABraOM"
    if "br__af_bin" in pairs.columns and "nonbr__af_bin" in pairs.columns:
        a = (pairs["br__af_bin"].astype("string") == "ausente").fillna(False)
        b = (pairs["nonbr__af_bin"].astype("string") == "ausente").fillna(False)
        out["mecanismo_bin_ausente"] = {"br_ausente": int(a.sum()), "ambos_ausentes": int((a & b).sum()),
                                        "concordancia_bin_ausente": float((a == b).mean())}
    return out


# ------------------------------------------------------------------------------------------------
# [C] sobreposicao com o Mosaic, [D] composicao, [E] ABraOM cru
# ------------------------------------------------------------------------------------------------


def mosaic_keys(path: Path) -> dict:
    import pandas as pd
    import pyarrow.parquet as pq

    names = set(pq.ParquetFile(path).schema_arrow.names)
    need = ["chrom", "pos_1based", "ref", "alt"]
    miss = [c for c in need if c not in names]
    if miss:
        raise SystemExit(f"{path}: faltam {miss}")
    cols = need + (["label_tier"] if "label_tier" in names else [])
    df = pd.read_parquet(path, columns=cols)
    keys = build_key(df["chrom"], df["pos_1based"], df["ref"], df["alt"])
    if "label_tier" not in df.columns:
        return {"todos": set(keys.dropna().astype(str))}
    tier = df["label_tier"].astype("string")
    return {str(t): set(keys[tier == t].dropna().astype(str)) for t in sorted(tier.dropna().unique())}


def audit_overlap(mosaic: dict, targets: dict) -> dict:
    return {tier: {name: len(keys & tset) for name, keys in targets.items()} | {"n_mosaic": len(tset)}
            for tier, tset in mosaic.items()}


def composition(df, key_col: str = "variant_key", type_col: str = "variant_type") -> dict:
    out: dict = {"n": int(len(df))}
    if type_col in df.columns:
        out["variant_type"] = {str(k): int(v) for k, v in df[type_col].value_counts(dropna=False).items()}
    if key_col in df.columns:
        snv = snv_from_key(df[key_col]).fillna(False).astype(bool)
        out["snv_pelo_key"] = int(snv.sum())
        out["nao_snv_pelo_key"] = int((~snv).sum())
    return out


def audit_abraom_tsv(path: Path, br_main, chunksize: int = 2_000_000) -> dict:
    import pandas as pd

    br_keys = set(br_main["variant_key"].astype(str))
    present = (dict(zip(br_main["variant_key"].astype(str), as_bool(br_main["abraom_present"])))
               if "abraom_present" in br_main.columns else {})
    n_rows = n_pos = 0
    min_pos = None
    below: Counter = Counter()
    hits: dict[str, float] = {}
    reader = pd.read_csv(path, sep="\t", usecols=["chrom", "pos", "ref", "alt", "af_abraom"],
                         dtype={"chrom": str, "ref": str, "alt": str}, chunksize=chunksize)
    for chunk in reader:
        n_rows += len(chunk)
        af = pd.to_numeric(chunk["af_abraom"], errors="coerce")
        pos_mask = (af > 0).fillna(False)
        n_pos += int(pos_mask.sum())
        if pos_mask.any():
            m = float(af[pos_mask].min())
            min_pos = m if min_pos is None else min(min_pos, m)
        for t in AF_THRESHOLDS:
            below[t] += int((pos_mask & (af < t)).sum())
        keys = build_key(chunk["chrom"], pd.to_numeric(chunk["pos"], errors="coerce"), chunk["ref"], chunk["alt"])
        sel = keys.isin(br_keys).fillna(False).to_numpy(dtype=bool)
        for k, a in zip(keys[sel].astype(str), af[sel]):
            hits[k] = float(a)
    in_tsv_absent_index = sorted(v for k, v in hits.items() if present and not present.get(k, False))
    in_index_absent_tsv = sum(1 for k, flag in present.items() if flag and k not in hits)
    return {
        "linhas_tsv": n_rows, "af_positiva": n_pos, "af_minima_positiva": min_pos,
        "af_positiva_abaixo_de": {str(t): below[t] for t in AF_THRESHOLDS},
        "t_br_n": len(br_keys), "t_br_no_tsv_cru": len(hits),
        "t_br_no_tsv_cru_mas_abraom_present_false": len(in_tsv_absent_index),
        "af_dessas": ({"min": in_tsv_absent_index[0], "mediana": in_tsv_absent_index[len(in_tsv_absent_index) // 2],
                       "max": in_tsv_absent_index[-1]} if in_tsv_absent_index else None),
        "t_br_abraom_present_mas_fora_do_tsv_cru": in_index_absent_tsv,
    }


# ------------------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    import pandas as pd

    args = parse_args(argv)
    excl = str(args.exclude_chrom).strip().lower().removeprefix("chr")
    report: dict = {"entradas": {k: (str(v) if v is not None else None) for k, v in vars(args).items()}}

    br, report["br_contagens"] = load_slice(args.br, excl)
    nonbr, report["nonbr_contagens"] = load_slice(args.nonbr, excl)
    pairs = pd.read_parquet(args.pairs)
    print(f"[audit] T_BR (sem chr{excl}): {len(br):,} · pool nonBR: {len(nonbr):,} · pares: {len(pairs):,}")

    print("\n[A] af_gnomad condicional ao ABraOM")
    report["A_af_gnomad_condicional"] = [audit_af_gnomad_conditional(br, "br_only"),
                                         audit_af_gnomad_conditional(nonbr, "nonbr_only")]
    for blk in report["A_af_gnomad_condicional"]:
        if "erro" in blk:
            print(f"    {blk['slice']}: {blk['erro']}")
            continue
        print(f"    {blk['slice']:<11} n={blk['n']:,}  abraom_present={blk['abraom_present']:,}  "
              f"af_gnomad preenchido={blk['af_gnomad_preenchido']:,}  zero/NA={blk['af_gnomad_zero_ou_na']:,}")
        print(f"                P(af_gnomad preenchido | fora do ABraOM) = {blk['p_af_preenchido_se_abraom_ausente']}")
        print(f"                -> {blk['veredito']}")

    print("\n[B] pareamento indireto por ABraOM (paragrafo 6.7)")
    b = audit_pair_abraom_concordance(pairs, nonbr)
    report["B_pareamento_abraom"] = b
    if "erro" in b:
        print(f"    {b['erro']}")
    else:
        for label in ("todos_os_estratos", "estratos_mistos"):
            blk = b[label]
            if blk.get("n"):
                print(f"    {label:<18} n={blk['n']:,}  observada={blk['concordancia_observada']:.3f}  "
                      f"esperada={blk['concordancia_esperada_sorteio_no_estrato']:.3f}  "
                      f"excesso={blk['excesso']:+.3f}  z={blk['z']:.1f}")
        print(f"    -> {b['veredito']}")

    print("\n[D] composicao por tipo de variante")
    report["D_composicao"] = {"t_br": composition(br), "t_nonbr_pareado": composition(
        pairs.rename(columns={"nonbr_variant_key": "variant_key", "nonbr_variant_type": "variant_type"}))}
    for name, blk in report["D_composicao"].items():
        print(f"    {name:<16} n={blk['n']:,}  snv={blk.get('snv_pelo_key')}  nao_snv={blk.get('nao_snv_pelo_key')}  "
              f"tipos={blk.get('variant_type')}")

    targets = {"t_br": set(br["variant_key"].astype(str)),
               "t_nonbr": set(pairs["nonbr_variant_key"].astype(str)) if "nonbr_variant_key" in pairs.columns else set()}
    if args.splits is not None:
        splits = pd.read_parquet(args.splits, columns=["variant_key", "split_within_gene"])
        for name, grp in splits.groupby("split_within_gene"):
            targets[f"split_{name}"] = set(grp["variant_key"].astype(str))
        report["D_composicao"]["splits"] = {str(n): composition(g) for n, g in splits.groupby("split_within_gene")}

    if args.mosaic_examples is not None:
        print("\n[C] sobreposicao com o Mosaic (a receita de extracao foi escolhida no gold)")
        report["C_sobreposicao_mosaic"] = audit_overlap(mosaic_keys(args.mosaic_examples), targets)
        for tier, blk in report["C_sobreposicao_mosaic"].items():
            print(f"    {tier:<10} " + "  ".join(f"{k}={v:,}" for k, v in blk.items()))

    if args.abraom_tsv is not None:
        print("\n[E] ABraOM cru (SABE-WGS-1171): piso de AF e cobertura do T_BR")
        e = audit_abraom_tsv(args.abraom_tsv, br)
        report["E_abraom_cru"] = e
        print(f"    linhas={e['linhas_tsv']:,}  AF minima positiva={e['af_minima_positiva']}  "
              f"abaixo de={e['af_positiva_abaixo_de']}")
        print(f"    T_BR no TSV cru: {e['t_br_no_tsv_cru']:,} de {e['t_br_n']:,} · "
              f"no cru mas abraom_present=False: {e['t_br_no_tsv_cru_mas_abraom_present_false']:,} "
              f"(AF {e['af_dessas']}) · abraom_present mas fora do cru: {e['t_br_abraom_present_mas_fora_do_tsv_cru']:,}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\n[audit] -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
