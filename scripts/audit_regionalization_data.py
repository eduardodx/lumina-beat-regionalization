#!/usr/bin/env python3
"""Auditoria READ-ONLY dos dados da regionalizacao R03, antes do redesenho sem adapter ClinVar e sem fusion.

Roda no notebook (pandas + pyarrow, este obrigatorio; sem GPU, sem S3). So escreve o JSON de saida.

POR QUE EXISTE
--------------
O redesenho (tirar o adapter ClinVar e a fusion, manter a adaptacao populacional) depende das colunas de
frequencia, do T_nonBR congelado e de qual conjunto de teste sera usado. Lendo o codigo, varios fatos que o
desenho precisa ficaram LIDOS mas nao MEDIDOS. Este script mede, sem supor:

  [A] af_gnomad condicional ao ABraOM. Em prepare_regional_clinvar_dataset.py o af_gnomad chega SO pelo
      merge (how="inner") com o indice ABraOM v2, e depois vai por left-join ao ClinVar. E o join so e
      TENTADO para SNVs (build_abraom_matches filtra is_snv): nao-SNV nunca tem abraom_present, af_abraom
      nem af_gnomad, por construcao. Por isso [A] sai separado para SNV e nao-SNV.
  [B] Pareamento associado ao ABraOM no T_nonBR. O matcher usa log10(af_gnomad.fillna(0)) na distancia.
      Compara a concordancia de abraom_present nos pares com a esperada se o matcher sorteasse dentro do
      estrato exato (gene, label, tipo), so nos estratos mistos. Os vereditos sao DESCRITIVOS.
  [C] Sobreposicao com o Mosaic por tier. A receita de extracao de embeddings foi escolhida no gold.
  [D] Composicao por tipo de variante E rotulo, no slice, nos membros BR dos pares e no controle. Uma
      campanha principal so-SNV depende de quantas BENIGNAS sobram.
  [E] (opcional, --abraom-tsv) ABraOM cru SABE-WGS-1171: AF minima, grade de AF e cobertura do T_BR.
  [F] (opcional, --mosaic-brazil-membership) Track `brazil` do Mosaic: dois estudos brasileiros ja
      pareados (br_clinical_evidence = consensus com instituicao brasileira; br_population_observed = gold
      presente no ABraOM), com contrato que PROIBE usar membership, controles e rotulos em treino, selecao
      ou calibracao. Mede quantos casos e controles de cada estudo caem nos nossos splits e nos T_BR/T_nonBR
      antigos. Se o track virar teste, essa sobreposicao tem de sair do treino antes de qualquer cabeca.

O QUE ESTE SCRIPT NAO PROVA
---------------------------
[B] mede EXCESSO de concordancia sobre um sorteio dentro do estrato. (1) O matcher real tambem usou
submitters, estrelas e consequencia, que podem correlacionar com a presenca no ABraOM por motivos
legitimos. (2) O z usa uma aproximacao binomial independente; o matcher sorteia sem reposicao e em
ordem gulosa, entao o z e indicativo, nao um teste. (3) Nada aqui da a DIRECAO de um efeito sobre a DiD.
O teste decisivo e re-parear preservando as demais regras e comparar os dois T_nonBR.
[E] O TSV do ABraOM NAO traz AC nem AN, so a AF. O denominador real so e inferido pela grade de AF
(quantas AF viram inteiro ao multiplicar pelo AN nominal). AN por sitio, cobertura e filtros precisam da
fonte original antes de definir bins de frequencia.

USO (notebook)
--------------
    PYTHONPATH="$WORK" "$PY" scripts/audit_regionalization_data.py \
        --br <br_only que o matcher usou> --nonbr <nonbr_only que o matcher usou> \
        --pairs ~/artifacts/fase0/t_nonbr_matched_soft.parquet \
        --splits ~/artifacts/fase0/clinvar_splits/clinvar_splits_combined.parquet \
        --mosaic-examples ~/mosaic-v1/pb_examples.parquet \
        --mosaic-brazil-membership ~/mosaic-v1/studies/brazil/membership.parquet \
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
ABRAOM_NOMINAL_AN = 2 * 1171   # SABE-WGS-1171 diploide, todos chamados -- so o nominal, ver [E]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--br", type=Path, required=True, help="slice br_only (a mesma que o matcher usou)")
    p.add_argument("--nonbr", type=Path, required=True, help="slice nonbr_only (a MESMA que o matcher usou)")
    p.add_argument("--pairs", type=Path, required=True, help="T_nonBR congelado (t_nonbr_matched_soft)")
    p.add_argument("--splits", type=Path, default=None, help="clinvar_splits_combined.parquet")
    p.add_argument("--mosaic-examples", type=Path, default=None, help="pb_examples.parquet do release Mosaic")
    p.add_argument("--mosaic-brazil-membership", type=Path, default=None,
                   help="studies/brazil/membership.parquet do release Mosaic (exige --mosaic-examples)")
    p.add_argument("--abraom-tsv", type=Path, default=None, help="SABE1171.Abraom.clean.tsv (opcional)")
    p.add_argument("--abraom-nominal-an", type=int, default=ABRAOM_NOMINAL_AN)
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


def snv_mask(frame, key_col: str = "variant_key"):
    """Mascara numpy de SNV derivada do variant_key (nao depende do dtype de is_snv)."""
    return snv_from_key(frame[key_col]).fillna(False).astype(bool).to_numpy()


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
# [B] pareamento associado ao ABraOM
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
                "excesso": obs - exp, "z_aproximado": (obs - exp) / se}

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
        "nota_incerteza": ("z por aproximacao binomial independente; o matcher sorteia sem reposicao e em "
                           "ordem gulosa. Indicativo, nao teste; nao informa a direcao de efeito na DiD."),
    }
    blk = out["estratos_mistos"]
    if blk.get("n", 0) == 0:
        out["veredito"] = "sem estratos mistos: nao ha como o matcher ter escolhido por ABraOM"
    elif blk["z_aproximado"] > 3 and blk["excesso"] > 0.05:
        out["veredito"] = ("EXCESSO acima do sorteio no estrato (descritivo, z aproximado): compativel com "
                           "pareamento associado a presenca no ABraOM, nao prova. Decisivo: re-parear "
                           "preservando as demais regras e comparar")
    elif blk["z_aproximado"] > 3:
        out["veredito"] = "EXCESSO pequeno acima do sorteio (<0.05, descritivo): registrar e comparar re-pareando"
    else:
        out["veredito"] = "sem excesso acima do sorteio no estrato (descritivo)"
    if "br__af_bin" in pairs.columns and "nonbr__af_bin" in pairs.columns:
        a = (pairs["br__af_bin"].astype("string") == "ausente").fillna(False)
        b = (pairs["nonbr__af_bin"].astype("string") == "ausente").fillna(False)
        out["mecanismo_bin_ausente"] = {"br_ausente": int(a.sum()), "ambos_ausentes": int((a & b).sum()),
                                        "concordancia_bin_ausente": float((a == b).mean())}
    return out


# ------------------------------------------------------------------------------------------------
# [C] sobreposicao com o Mosaic, [D] composicao, [E] ABraOM cru, [F] track brazil
# ------------------------------------------------------------------------------------------------


def mosaic_example_keys(path: Path):
    """pb_examples do Mosaic com a chave regional (chrom:pos:ref:alt, chrom sem prefixo)."""
    import pandas as pd
    import pyarrow.parquet as pq

    names = set(pq.ParquetFile(path).schema_arrow.names)
    need = ["variant_id", "chrom", "pos_1based", "ref", "alt"]
    miss = [c for c in need if c not in names]
    if miss:
        raise SystemExit(f"{path}: faltam {miss}")
    cols = need + (["label_tier"] if "label_tier" in names else [])
    df = pd.read_parquet(path, columns=cols)
    df["key"] = build_key(df["chrom"], df["pos_1based"], df["ref"], df["alt"])
    return df


def mosaic_keys(path: Path) -> dict:
    df = mosaic_example_keys(path)
    if "label_tier" not in df.columns:
        return {"todos": set(df["key"].dropna().astype(str))}
    tier = df["label_tier"].astype("string")
    return {str(t): set(df.loc[tier == t, "key"].dropna().astype(str)) for t in sorted(tier.dropna().unique())}


def audit_overlap(mosaic: dict, targets: dict) -> dict:
    return {tier: {name: len(keys & tset) for name, keys in targets.items()} | {"n_mosaic": len(tset)}
            for tier, tset in mosaic.items()}


def audit_mosaic_brazil_overlap(membership_path: Path, examples_path: Path, targets: dict) -> dict:
    """Membros do track brazil (por estudo e papel) que caem nos nossos splits e testes."""
    import pandas as pd
    import pyarrow.parquet as pq

    names = set(pq.ParquetFile(membership_path).schema_arrow.names)
    need = ["variant_id", "study_id", "member_role"]
    miss = [c for c in need if c not in names]
    if miss:
        raise SystemExit(f"{membership_path}: faltam {miss}")
    mem = pd.read_parquet(membership_path, columns=need)
    ex = mosaic_example_keys(examples_path)[["variant_id", "key"]]
    mem = mem.merge(ex, on="variant_id", how="left")
    out: dict = {"membros": int(len(mem)), "sem_chave_no_pb_examples": int(mem["key"].isna().sum()), "grupos": {}}
    train_like = [name for name in targets if name.startswith("split_")]
    contaminated = []
    for (study, role), grp in mem.groupby(["study_id", "member_role"]):
        keys = set(grp["key"].dropna().astype(str))
        blk = {"n": int(len(grp))} | {name: len(keys & tset) for name, tset in targets.items()}
        out["grupos"][f"{study}/{role}"] = blk
        hits = {name: blk[name] for name in train_like if blk.get(name)}
        if hits:
            contaminated.append(f"{study}/{role} {hits}")
    out["veredito"] = (
        "membros do track brazil caem nos nossos splits: se o track for usado como teste, excluir antes de "
        "treinar (o contrato do Mosaic proibe treino, selecao e calibracao neles) -> " + "; ".join(contaminated)
        if contaminated else "nenhum membro do track brazil nos splits informados"
    )
    return out


def composition(df, key_col: str = "variant_key", type_col: str = "variant_type", label_col: str = "label") -> dict:
    """Contagens por tipo e por rotulo. O lado benigno e o que limita o poder de uma campanha so-SNV."""
    import pandas as pd

    out: dict = {"n": int(len(df))}
    lab = pd.to_numeric(df[label_col], errors="coerce") if label_col in df.columns else None
    if type_col in df.columns:
        types = df[type_col].astype("string").fillna("NA")
        out["variant_type"] = {str(k): int(v) for k, v in types.value_counts().items()}
        if lab is not None:
            out["variant_type_x_label"] = {
                str(t): {"P": int(((types == t) & (lab == 1)).sum()), "B": int(((types == t) & (lab == 0)).sum())}
                for t in types.unique()
            }
    if key_col in df.columns:
        snv = snv_from_key(df[key_col]).fillna(False).astype(bool)
        out["snv_pelo_key"] = int(snv.sum())
        out["nao_snv_pelo_key"] = int((~snv).sum())
        if lab is not None:
            out["snv_P"] = int((snv & (lab == 1)).sum())
            out["snv_B"] = int((snv & (lab == 0)).sum())
            out["nao_snv_P"] = int((~snv & (lab == 1)).sum())
            out["nao_snv_B"] = int((~snv & (lab == 0)).sum())
    return out


def audit_abraom_tsv(path: Path, br_main, chunksize: int = 2_000_000, nominal_an: int = ABRAOM_NOMINAL_AN) -> dict:
    import pandas as pd

    br_keys = set(br_main["variant_key"].astype(str))
    present = (dict(zip(br_main["variant_key"].astype(str), as_bool(br_main["abraom_present"])))
               if "abraom_present" in br_main.columns else {})
    n_rows = n_pos = lattice_ok = 0
    min_pos = None
    below: Counter = Counter()
    small_afs: Counter = Counter()
    hits: dict[str, float] = {}
    reader = pd.read_csv(path, sep="\t", usecols=["chrom", "pos", "ref", "alt", "af_abraom"],
                         dtype={"chrom": str, "ref": str, "alt": str}, chunksize=chunksize)
    for chunk in reader:
        n_rows += len(chunk)
        af = pd.to_numeric(chunk["af_abraom"], errors="coerce")
        pos_mask = (af > 0).fillna(False)
        n_pos += int(pos_mask.sum())
        if pos_mask.any():
            pos_af = af[pos_mask]
            m = float(pos_af.min())
            min_pos = m if min_pos is None else min(min_pos, m)
            k = pos_af * nominal_an
            lattice_ok += int(((k - k.round()).abs() < 0.01).sum())
            small_afs.update(pos_af[pos_af < 0.005].round(7).tolist())
        for t in AF_THRESHOLDS:
            below[t] += int((pos_mask & (af < t)).sum())
        keys = build_key(chunk["chrom"], pd.to_numeric(chunk["pos"], errors="coerce"), chunk["ref"], chunk["alt"])
        sel = keys.isin(br_keys).fillna(False).to_numpy(dtype=bool)
        for key, a in zip(keys[sel].astype(str), af[sel]):
            hits[key] = float(a)
    in_tsv_absent_index = sorted(v for key, v in hits.items() if present and not present.get(key, False))
    in_index_absent_tsv = sum(1 for key, flag in present.items() if flag and key not in hits)
    return {
        "linhas_tsv": n_rows, "af_positiva": n_pos, "af_minima_positiva": min_pos,
        "af_positiva_abaixo_de": {str(t): below[t] for t in AF_THRESHOLDS},
        "grade_af": {
            "an_nominal": nominal_an,
            "frac_af_positiva_multiplo_de_1_sobre_an_nominal": (lattice_ok / n_pos) if n_pos else None,
            "menores_af_distintas": [[v, c] for v, c in sorted(small_afs.items())[:10]],
            "nota": "o TSV nao traz AC/AN; AN real, cobertura e filtros precisam da fonte original",
        },
        "t_br_n": len(br_keys), "t_br_no_tsv_cru": len(hits),
        "t_br_no_tsv_cru_mas_abraom_present_false": len(in_tsv_absent_index),
        "af_dessas": ({"min": in_tsv_absent_index[0], "mediana": in_tsv_absent_index[len(in_tsv_absent_index) // 2],
                       "max": in_tsv_absent_index[-1]} if in_tsv_absent_index else None),
        "t_br_abraom_present_mas_fora_do_tsv_cru": in_index_absent_tsv,
    }


# ------------------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    import pandas as pd
    import pyarrow.parquet as pq

    args = parse_args(argv)
    excl = str(args.exclude_chrom).strip().lower().removeprefix("chr")
    report: dict = {"entradas": {k: (str(v) if v is not None else None) for k, v in vars(args).items()}}

    br, report["br_contagens"] = load_slice(args.br, excl)
    nonbr, report["nonbr_contagens"] = load_slice(args.nonbr, excl)
    pairs = pd.read_parquet(args.pairs)
    print(f"[audit] T_BR (sem chr{excl}): {len(br):,} · pool nonBR: {len(nonbr):,} · pares: {len(pairs):,}")

    print("\n[A] af_gnomad condicional ao ABraOM (separado por SNV: o join do ABraOM so tenta SNV)")
    blocks_a = []
    for frame, name in ((br, "br_only"), (nonbr, "nonbr_only")):
        snv = snv_mask(frame)
        blocks_a.append(audit_af_gnomad_conditional(frame[snv], f"{name}/SNV"))
        blocks_a.append(audit_af_gnomad_conditional(frame[~snv], f"{name}/nao-SNV"))
    report["A_af_gnomad_condicional"] = blocks_a
    for blk in blocks_a:
        if "erro" in blk:
            print(f"    {blk['slice']}: {blk['erro']}")
            continue
        print(f"    {blk['slice']:<20} n={blk['n']:,}  abraom_present={blk['abraom_present']:,}  "
              f"af_gnomad preenchido={blk['af_gnomad_preenchido']:,}  zero/NA={blk['af_gnomad_zero_ou_na']:,}")
        print(f"                         P(af_gnomad preenchido | fora do ABraOM) = "
              f"{blk['p_af_preenchido_se_abraom_ausente']}  -> {blk['veredito']}")
        if blk["slice"].endswith("nao-SNV") and blk["n"] > 0 and blk["abraom_present"] == 0:
            print("                         (estrutural: build_abraom_matches so tenta SNV; nao-SNV nunca entra no ABraOM)")

    print("\n[B] pareamento associado ao ABraOM (paragrafo 6.7) -- descritivo")
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
                      f"excesso={blk['excesso']:+.3f}  z~{blk['z_aproximado']:.1f}")
        print(f"    -> {b['veredito']}")
        print(f"    ({b['nota_incerteza']})")

    print("\n[D] composicao por tipo e rotulo (poder de uma campanha principal so-SNV)")
    pair_br = pairs.rename(columns={"br_variant_key": "variant_key", "br_variant_type": "variant_type",
                                    "br_label": "label"})
    pair_nb = pairs.rename(columns={"nonbr_variant_key": "variant_key", "nonbr_variant_type": "variant_type",
                                    "nonbr_label": "label"})
    report["D_composicao"] = {"t_br_slice": composition(br), "t_br_pareado": composition(pair_br),
                              "t_nonbr_pareado": composition(pair_nb)}
    for name, blk in report["D_composicao"].items():
        print(f"    {name:<16} n={blk['n']:,}  SNV P/B={blk.get('snv_P')}/{blk.get('snv_B')}  "
              f"nao-SNV P/B={blk.get('nao_snv_P')}/{blk.get('nao_snv_B')}")

    targets = {"t_br_slice": set(br["variant_key"].astype(str))}
    for side in ("br", "nonbr"):
        col = f"{side}_variant_key"
        if col in pairs.columns:
            targets[f"t_{side}_pareado"] = set(pairs[col].astype(str))
    if args.splits is not None:
        names = set(pq.ParquetFile(args.splits).schema_arrow.names)
        cols = [c for c in ("variant_key", "split_within_gene", "label") if c in names]
        splits = pd.read_parquet(args.splits, columns=cols)
        for name, grp in splits.groupby("split_within_gene"):
            targets[f"split_{name}"] = set(grp["variant_key"].astype(str))
        report["D_composicao"]["splits"] = {str(n): composition(g) for n, g in splits.groupby("split_within_gene")}

    if args.mosaic_examples is not None:
        print("\n[C] sobreposicao com o Mosaic por tier (a receita de extracao foi escolhida no gold)")
        report["C_sobreposicao_mosaic"] = audit_overlap(mosaic_keys(args.mosaic_examples), targets)
        for tier, blk in report["C_sobreposicao_mosaic"].items():
            print(f"    {tier:<10} " + "  ".join(f"{k}={v:,}" for k, v in blk.items()))

    if args.abraom_tsv is not None:
        print("\n[E] ABraOM cru (SABE-WGS-1171): piso, grade de AF e cobertura do T_BR")
        e = audit_abraom_tsv(args.abraom_tsv, br, nominal_an=args.abraom_nominal_an)
        report["E_abraom_cru"] = e
        g = e["grade_af"]
        print(f"    linhas={e['linhas_tsv']:,}  AF minima positiva={e['af_minima_positiva']}  "
              f"abaixo de={e['af_positiva_abaixo_de']}")
        print(f"    grade: fracao multipla de 1/{g['an_nominal']} = {g['frac_af_positiva_multiplo_de_1_sobre_an_nominal']}  "
              f"menores AF distintas={g['menores_af_distintas'][:5]}  ({g['nota']})")
        print(f"    T_BR no TSV cru: {e['t_br_no_tsv_cru']:,} de {e['t_br_n']:,} · "
              f"no cru mas abraom_present=False: {e['t_br_no_tsv_cru_mas_abraom_present_false']:,} "
              f"(AF {e['af_dessas']}) · abraom_present mas fora do cru: {e['t_br_abraom_present_mas_fora_do_tsv_cru']:,}")

    if args.mosaic_brazil_membership is not None:
        print("\n[F] track brazil do Mosaic: membros nos nossos splits e testes")
        if args.mosaic_examples is None:
            print("    --mosaic-brazil-membership exige --mosaic-examples (a chave vem do pb_examples)")
        else:
            f = audit_mosaic_brazil_overlap(args.mosaic_brazil_membership, args.mosaic_examples, targets)
            report["F_mosaic_brazil"] = f
            print(f"    membros={f['membros']:,}  sem chave no pb_examples={f['sem_chave_no_pb_examples']:,}")
            for grupo, blk in sorted(f["grupos"].items()):
                print(f"    {grupo:<42} " + "  ".join(f"{k}={v:,}" for k, v in blk.items()))
            print(f"    -> {f['veredito']}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\n[audit] -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
