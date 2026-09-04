#!/usr/bin/env python3
"""Fase 0 da sonda de embedding: le o release Mosaic v1 e monta o manifesto de variantes.

Roda no NOTEBOOK, em CPU (sem torch, sem GPU). Faz cinco coisas, nessa ordem, e para na primeira
que falhar -- a ideia e que nenhuma suposicao do plano chegue na GPU sem ter sido conferida:

  1. **Preflight**: os arquivos do release existem e tem as colunas que o codigo do Mosaic declara.
  2. **Cross-check de FASTA** (BLOQUEANTE): o `~/hg38/hg38.fa` do notebook e o
     `GRCh38.p14.chr1-22.fa` com que o Mosaic validou o REF sao o mesmo primario? Amostra N
     variantes e exige 100% de REF batendo. Se falhar, a sonda inteira estaria lendo o genoma
     errado -- melhor descobrir aqui do que depois de milhares de forwards.
  3. **Censo de sitios multialelicos**: quantos sitios tem >=2 ALT, >=3 ALT, quantos tem ALTs de
     consequencia diferente, e -- o que decide o experimento E2 -- quantos tem um ALT patogenico
     e outro benigno no MESMO sitio.
  4. **Selecao**: braco `curated` (sitios multialelicos, prioridade P-vs-B) + braco `statistical`
     (gold estratificado por painel x label).
  5. **Validacao de janela**: para cada variante escolhida, tenta construir as 5 janelas centradas
     e as 3 casadas. Isso e o que pega o layout `matched`, que se estende ate 30.720 bp downstream
     -- muito alem dos +-16.383 que o `sequence_eligible` do Mosaic conferiu.

Pandas so aparece na borda (carga e escrita). Censo, selecao e validacao operam sobre `Variant`
(NamedTuple) para serem testaveis sem ambiente cientifico -- ver
`tests/test_probe_build_manifest.py`, que roda em stdlib puro.

USO
---
    PYTHONPATH="$WORK" "$PY" scripts/probe_build_manifest.py \
        --release-root ~/mosaic-v1 \
        --fasta ~/hg38/hg38.fa \
        --out-dir ~/probe

SAIDA
-----
    probe_manifest.parquet   uma linha por (variante, arm) -- entrada da Fase B (GPU)
    probe_census.json        o censo completo, para revisao antes de gastar GPU
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, NamedTuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.embedding_probe.windows import (  # noqa: E402
    CONSUMER_WINDOWS,
    EXPLORATORY_WINDOWS,
    WindowError,
    build_window,
    matched_focal_index,
)

ALL_WINDOWS: tuple[int, ...] = tuple(sorted(EXPLORATORY_WINDOWS + CONSUMER_WINDOWS))
SEED = 20260903

# Colunas que o codigo do Mosaic declara (examples.py::EXAMPLES_SCHEMA, panels.py::PANELS_SCHEMA,
# annotations/schema.py, partitions.py::PARTITIONS_SCHEMA). Conferidas no preflight em vez de
# assumidas -- se o release publicado divergir do codigo, queremos saber na primeira linha.
REQUIRED: dict[str, list[str]] = {
    "pb_examples.parquet": [
        "variant_id", "chrom", "pos_1based", "ref", "alt",
        "binary_label", "label_tier", "sequence_eligible", "br_lab_any",
    ],
    "pb_panels.parquet": ["variant_id", "primary_panel", "panel_role"],
    "pb_annotations.parquet": ["variant_id", "region_class", "phylop_241way", "mane_gene", "gnomad_af_bin"],
    "pb_partitions.parquet": ["variant_id", "overlap_cluster_id", "resolved_gene"],
}


class Variant(NamedTuple):
    """Uma variante do release, ja com os joins resolvidos."""

    variant_id: str
    chrom: str
    pos_1based: int
    ref: str
    alt: str
    binary_label: int
    label_tier: str
    primary_panel: str
    panel_role: str
    region_class: str
    phylop_241way: float
    mane_gene: str
    resolved_gene: str
    gnomad_af_bin: str
    overlap_cluster_id: str
    br_lab_any: bool


SiteKey = tuple[str, int, str]
Fetcher = Callable[[str, int, int], str]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--release-root", type=Path, required=True, help="raiz do release Mosaic v1 baixado")
    p.add_argument("--fasta", type=Path, required=True, help="hg38.fa do notebook")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--fasta-check-n", type=int, default=5000, help="variantes amostradas no cross-check")
    p.add_argument("--n-curated-sites", type=int, default=20, help="sitios multialelicos para o braco curated")
    p.add_argument("--n-statistical", type=int, default=2000, help="variantes gold do braco statistical")
    p.add_argument("--seed", type=int, default=SEED)
    return p.parse_args(argv)


# ------------------------------------------------------------------------------------------------
# 1. Preflight / carga (unica parte que fala pandas)
# ------------------------------------------------------------------------------------------------


def preflight(release_root: Path) -> tuple[list[Variant], dict]:
    print("\n[1/5] PREFLIGHT")
    import pandas as pd
    import pyarrow.parquet as pq

    frames = {}
    for name, cols in REQUIRED.items():
        path = release_root / name
        if not path.is_file():
            raise SystemExit(
                f"FALTA {path}\n"
                f"  conteudo de {release_root}: {sorted(p.name for p in release_root.glob('*')) or '(vazio)'}\n"
                f"  esperado o release plano: s3://croma-bioai-lumina-releases-us-east-2/benchmarks/mosaic/v1/"
            )
        # Confere o schema ANTES de pedir o subconjunto de colunas: `read_parquet(columns=...)`
        # levantaria um erro do pyarrow que esconde qual coluna divergiu do codigo do Mosaic.
        available = set(pq.read_schema(path).names)
        missing = [c for c in cols if c not in available]
        if missing:
            raise SystemExit(f"{name}: faltam colunas {missing}\n  presentes: {sorted(available)}")
        df = pd.read_parquet(path, columns=cols)
        frames[name] = df
        print(f"  [ok] {name:<26} {len(df):>8,} linhas")

    identity: dict = {}
    manifest_path = release_root / "release.manifest.json"
    if manifest_path.is_file():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        identity = {k: v for k, v in payload.items() if k in ("release_id", "release_hash", "clinvar_release")}
        print(f"  [ok] release identity: {identity or '(campos esperados ausentes)'}")
    else:
        print("  [!!] release.manifest.json ausente -- proveniencia do release nao registrada")

    df = frames["pb_examples.parquet"]
    for name in ("pb_panels.parquet", "pb_annotations.parquet", "pb_partitions.parquet"):
        df = df.merge(frames[name], on="variant_id", validate="one_to_one")
    assert df["variant_id"].is_unique, "variant_id duplicado apos os joins"
    print(f"  [ok] universo unificado: {len(df):,} variantes")

    eligible_df = df[df["sequence_eligible"]]
    print(f"  [ok] sequence_eligible:   {len(eligible_df):,} ({len(eligible_df) / len(df):.1%})")
    print(f"       por tier: {dict(Counter(eligible_df['label_tier']))}")
    if eligible_df.empty:
        raise SystemExit("nenhuma variante sequence_eligible")

    fields = Variant._fields
    records = [
        Variant(**{f: row[f] for f in fields})
        for row in eligible_df[list(fields)].to_dict("records")
    ]
    return records, identity


def open_fasta(path: Path) -> tuple[object, Fetcher]:
    from pyfaidx import Fasta

    fasta = Fasta(str(path), as_raw=True, sequence_always_upper=True)

    def fetch(chrom: str, start: int, end: int) -> str:
        if start < 0 or chrom not in fasta:
            return ""
        return str(fasta[chrom][start:end])

    return fasta, fetch


# ------------------------------------------------------------------------------------------------
# 2. Cross-check de FASTA
# ------------------------------------------------------------------------------------------------


def check_fasta(records: list[Variant], fetch: Fetcher, contigs: set[str], n: int, seed: int) -> dict:
    """BLOQUEANTE: o hg38 do notebook tem que concordar com o REF validado pelo Mosaic."""
    print(f"\n[2/5] CROSS-CHECK DE FASTA (hg38 do notebook vs GRCh38.p14 do Mosaic, n={n:,})")
    chroms = sorted({r.chrom for r in records})
    missing = [c for c in chroms if c not in contigs]
    if missing:
        raise SystemExit(
            f"contigs ausentes no FASTA: {missing}\n"
            f"  o Mosaic usa nomes UCSC (chr1..chr22); o FASTA tem, por exemplo: {sorted(contigs)[:5]}"
        )
    print(f"  [ok] {len(chroms)} contigs do Mosaic presentes no FASTA")

    sample = random.Random(seed).sample(records, min(n, len(records)))
    mismatches = [
        {"variant_id": r.variant_id, "chrom": r.chrom, "pos": r.pos_1based,
         "mosaic_ref": r.ref, "fasta": fetch(r.chrom, r.pos_1based - 1, r.pos_1based)}
        for r in sample
        if fetch(r.chrom, r.pos_1based - 1, r.pos_1based) != r.ref
    ]
    rate = 1 - len(mismatches) / len(sample)
    print(f"  REF batendo: {rate:.6%} ({len(sample) - len(mismatches):,}/{len(sample):,})")
    if mismatches:
        for m in mismatches[:5]:
            print(f"    MISMATCH {m['chrom']}:{m['pos']} mosaic={m['mosaic_ref']} fasta={m['fasta']!r}")
        raise SystemExit(
            f"\n{len(mismatches)} REF divergentes -- o hg38 do notebook NAO e o primario do GRCh38.p14.\n"
            "Use o FASTA do proprio release (reference/GRCh38.p14.chr1-22.fa) antes de seguir."
        )
    print("  [ok] os primarios chr1-22 sao identicos; seguimos com o hg38 do notebook")
    return {"n_checked": len(sample), "match_rate": 1.0, "n_contigs": len(chroms)}


# ------------------------------------------------------------------------------------------------
# 3. Censo
# ------------------------------------------------------------------------------------------------


def group_by_site(records: list[Variant]) -> dict[SiteKey, list[Variant]]:
    """Agrupa por (chrom, pos, ref) -- a mesma unidade que `mosaic.clusters` funde primeiro."""
    by_site: dict[SiteKey, list[Variant]] = defaultdict(list)
    for record in records:
        by_site[(record.chrom, record.pos_1based, record.ref)].append(record)
    return dict(by_site)


def census(by_site: dict[SiteKey, list[Variant]], *, verbose: bool = True) -> dict:
    """Censo dos sitios multialelicos -- decide se o experimento E2 (P vs B) existe."""
    def say(msg: str) -> None:
        if verbose:
            print(msg)

    say("\n[3/5] CENSO DE SITIOS MULTIALELICOS")
    multi = {k: v for k, v in by_site.items() if len(v) >= 2}
    stats: dict = {
        "n_sites": len(by_site),
        "n_sites_multiallelic": len(multi),
        "n_sites_3plus": sum(1 for v in multi.values() if len(v) >= 3),
    }
    say(f"  sitios distintos (chrom,pos,ref): {len(by_site):,}")
    say(f"  sitios com >=2 ALT:               {len(multi):,}")
    say(f"  sitios com >=3 ALT:               {stats['n_sites_3plus']:,}")

    def summarize(keep, tag: str) -> dict:
        sites = {}
        for key, rows in by_site.items():
            kept = [r for r in rows if keep(r.label_tier)]
            if len(kept) >= 2:
                sites[key] = kept
        pb = {k: v for k, v in sites.items() if {0, 1} <= {int(r.binary_label) for r in v}}
        say(f"  [{tag}] >=2 ALT: {len(sites):>6,}   com P *e* B no mesmo sitio: {len(pb):>6,}")
        return {"n_multiallelic": len(sites), "n_with_both_classes": len(pb)}

    stats["gold"] = summarize(lambda t: t == "gold", "gold          ")
    stats["consensus"] = summarize(lambda t: t == "consensus", "consensus     ")
    stats["any_tier"] = summarize(lambda t: True, "gold+consensus")

    # Mesmo sitio, ALTs com CONSEQUENCIA diferente (ex.: missense vs synonymous). Contexto e
    # posicao 100% identicos, muda so o efeito funcional do alelo -- segundo melhor contraste
    # depois do P-vs-B, e util mesmo se o P-vs-B nao existir.
    het = {k: v for k, v in multi.items() if len({r.primary_panel for r in v}) >= 2}
    stats["n_sites_panel_heterogeneous"] = len(het)
    say(f"  sitios com >=2 paineis distintos entre os ALTs: {len(het):,}")

    if stats["gold"]["n_with_both_classes"] == 0:
        say("\n  >> RAMIFICACAO: nenhum sitio GOLD com P e B juntos.")
        if stats["any_tier"]["n_with_both_classes"] > 0:
            say("     Ha em gold+consensus -- E2 usa esse tier, declarado no relatorio.")
        else:
            say("     Nao ha em tier nenhum -- E2 cai para 3 alelos quaisquer (o pedido original).")
    return stats


# ------------------------------------------------------------------------------------------------
# 4. Selecao
# ------------------------------------------------------------------------------------------------


def site_priority(rows_at_site: list[Variant]) -> tuple:
    """Ordem do braco `curated`. Menor e melhor."""
    tiers = {r.label_tier for r in rows_at_site}
    labels = {int(r.binary_label) for r in rows_at_site}
    panels = {r.primary_panel for r in rows_at_site}
    both_classes = {0, 1} <= labels
    all_gold = tiers == {"gold"}
    return (
        0 if (both_classes and all_gold) else 1 if both_classes else 2,  # P-vs-B primeiro
        0 if len(panels) >= 2 else 1,                                    # consequencias diferentes
        -len(rows_at_site),                                              # depois mais ALTs
        0 if all_gold else 1,                                            # depois gold puro
    )


def choose_curated_sites(
    by_site: dict[SiteKey, list[Variant]], budget: int
) -> list[tuple[SiteKey, list[Variant]]]:
    """Tres passes, para nao entregar menos sitios que o pedido em silencio se os paineis forem
    pouco diversos: (1) metade do orcamento por prioridade pura, (2) um sitio por painel ainda nao
    coberto, (3) completa o orcamento por prioridade."""
    multi = sorted(
        ((k, v) for k, v in by_site.items() if len(v) >= 2),
        key=lambda kv: (site_priority(kv[1]), kv[0]),
    )
    chosen: list[tuple[SiteKey, list[Variant]]] = []
    taken: set[SiteKey] = set()

    def take(key: SiteKey, rows: list[Variant]) -> None:
        chosen.append((key, rows))
        taken.add(key)

    for key, rows in multi:
        if len(chosen) >= budget // 2:
            break
        take(key, rows)
    panels_seen = {r.primary_panel for _, rows in chosen for r in rows}
    for key, rows in multi:
        if len(chosen) >= budget:
            break
        panels = {r.primary_panel for r in rows}
        if key not in taken and panels - panels_seen:
            take(key, rows)
            panels_seen |= panels
    for key, rows in multi:
        if len(chosen) >= budget:
            break
        if key not in taken:
            take(key, rows)
    return chosen


def select(
    records: list[Variant], by_site: dict[SiteKey, list[Variant]], args, *, verbose: bool = True
) -> tuple[list[dict], dict]:
    """Braco `curated` (sitios multialelicos) + braco `statistical` (gold estratificado)."""
    def say(msg: str) -> None:
        if verbose:
            print(msg)

    say("\n[4/5] SELECAO")
    rng = random.Random(args.seed)
    rows: list[dict] = []
    emitted: set[tuple[str, str]] = set()

    def emit(v: Variant, arm: str, site_key: str, site_rank: int | None = None) -> None:
        if (v.variant_id, arm) in emitted:
            return
        emitted.add((v.variant_id, arm))
        rows.append({
            **v._asdict(),
            "arm": arm,
            "site_key": site_key,
            "site_rank": site_rank,
            "binary_label": int(v.binary_label),
            "br_lab_any": bool(v.br_lab_any),
        })

    chosen_sites = choose_curated_sites(by_site, args.n_curated_sites)
    for rank, (key, rows_at_site) in enumerate(chosen_sites):
        site_key = f"{key[0]}:{key[1]}:{key[2]}"
        for v in rows_at_site:
            emit(v, "curated", site_key, rank)
    curated_panels = sorted({r.primary_panel for _, rs in chosen_sites for r in rs})
    say(f"  curated:     {len(chosen_sites)} sitios, {sum(len(v) for _, v in chosen_sites)} alelos"
        f"  (paineis: {curated_panels})")

    strata: dict[tuple[str, int], list[Variant]] = defaultdict(list)
    for v in records:
        if v.label_tier == "gold":
            strata[(v.primary_panel, int(v.binary_label))].append(v)
    per = max(1, args.n_statistical // max(1, len(strata)))
    for key in sorted(strata):
        bucket = list(strata[key])
        rng.shuffle(bucket)
        for v in bucket[:per]:
            emit(v, "statistical", f"{v.chrom}:{v.pos_1based}:{v.ref}")
    n_stat = sum(1 for r in rows if r["arm"] == "statistical")
    say(f"  statistical: {n_stat} variantes gold em {len(strata)} estratos (painel x label, "
        f"alvo {per}/estrato)")

    # O controle "mesma troca, loci diferentes" e ANALISE POST-HOC do braco statistical -- nao
    # precisa de forward extra. Aqui so conferimos que ele vai ter poder.
    subs = Counter(f"{r['ref']}>{r['alt']}" for r in rows if r["arm"] == "statistical")
    say(f"  controle mesma-troca (post-hoc): {len(subs)} tipos; top5 "
        f"{[f'{k}={v}' for k, v in subs.most_common(5)]}")
    thin = sorted(k for k, v in subs.items() if v < 30)
    if thin:
        say(f"    [!] tipos com n<30 (controle fraco nesses): {thin}")
    return rows, {
        "n_curated_sites": len(chosen_sites),
        "curated_panels": curated_panels,
        "n_statistical": n_stat,
        "n_strata": len(strata),
        "substitution_counts": dict(subs),
        "substitutions_underpowered": thin,
    }


# ------------------------------------------------------------------------------------------------
# 5. Validacao de janelas
# ------------------------------------------------------------------------------------------------


def window_plan() -> list[tuple[int, int | None]]:
    """As janelas que a Fase B vai extrair: 5 centradas + 3 casadas."""
    matched = matched_focal_index(CONSUMER_WINDOWS)
    return [(bp, None) for bp in ALL_WINDOWS] + [(bp, matched) for bp in CONSUMER_WINDOWS]


def validate_windows(rows: list[dict], fetch: Fetcher, *, verbose: bool = True) -> dict:
    """Confere que TODAS as janelas pedidas existem de fato -- inclusive as `matched`, que saem
    do envelope de +-16.383 que o `sequence_eligible` do Mosaic validou."""
    def say(msg: str) -> None:
        if verbose:
            print(msg)

    say("\n[5/5] VALIDACAO DE JANELAS")
    plan = window_plan()
    labels = [f"{bp}/{'centered' if f is None else 'matched'}" for bp, f in plan]
    failures: Counter = Counter()
    ok_by_variant: dict[str, list[str]] = {}
    for row in rows:
        vid = row["variant_id"]
        if vid in ok_by_variant:
            continue
        good: list[str] = []
        for (window_bp, focal), label in zip(plan, labels):
            try:
                build_window(fetch, chrom=row["chrom"], pos_1based=row["pos_1based"],
                             ref=row["ref"], alt=row["alt"], window_bp=window_bp, focal_index=focal)
                good.append(label)
            except WindowError as exc:
                failures[f"{label}/{exc.reason}"] += 1
        ok_by_variant[vid] = good

    # Guardamos as janelas validas POR variante em vez de um booleano tudo-ou-nada: uma variante
    # que so falha na 32k/matched ainda serve para as outras sete. A Fase B le esta coluna e pula
    # apenas as combinacoes ausentes.
    for row in rows:
        good = ok_by_variant[row["variant_id"]]
        row["windows_ok"] = ",".join(good)
        row["n_windows_ok"] = len(good)
        row["all_windows_ok"] = len(good) == len(plan)

    n_all = sum(1 for v in ok_by_variant.values() if len(v) == len(plan))
    n_none = sum(1 for v in ok_by_variant.values() if not v)
    per_window = Counter(label for good in ok_by_variant.values() for label in good)
    say(f"  variantes com as {len(plan)} janelas validas: {n_all:,}/{len(ok_by_variant):,}"
        f"   (sem nenhuma janela: {n_none:,})")
    say("  cobertura por janela:")
    for label in labels:
        say(f"    {label:<20} {per_window.get(label, 0):>6,}/{len(ok_by_variant):,}")
    if failures:
        say("  falhas por (janela/layout/motivo):")
        for key, count in failures.most_common():
            say(f"    {key:<40} {count:>6,}")
        say("  (esperado sobretudo em 32768/matched: 30.720 bp downstream saem do envelope 32k)")
    else:
        say("  [ok] nenhuma falha")
    return {
        "n_variants_all_ok": n_all,
        "n_variants_no_window": n_none,
        "n_variants": len(ok_by_variant),
        "coverage_per_window": {label: per_window.get(label, 0) for label in labels},
        "failures": dict(failures),
        "plan": labels,
    }


# ------------------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    records, identity = preflight(args.release_root)
    fasta, fetch = open_fasta(args.fasta)
    fasta_report = check_fasta(records, fetch, set(fasta.keys()), args.fasta_check_n, args.seed)
    by_site = group_by_site(records)
    census_stats = census(by_site)
    rows, selection_stats = select(records, by_site, args)
    window_report = validate_windows(rows, fetch)

    import pandas as pd

    manifest = pd.DataFrame(rows)
    out_parquet = args.out_dir / "probe_manifest.parquet"
    manifest.to_parquet(out_parquet, index=False)

    report = {
        "seed": args.seed,
        "release_identity": identity,
        "release_root": str(args.release_root),
        "fasta": str(args.fasta),
        "windows_centered": list(ALL_WINDOWS),
        "windows_matched": list(CONSUMER_WINDOWS),
        "matched_focal_index": matched_focal_index(CONSUMER_WINDOWS),
        "fasta_check": fasta_report,
        "census": census_stats,
        "selection": selection_stats,
        "window_validation": window_report,
        "n_manifest_rows": len(manifest),
        "n_manifest_variants": int(manifest["variant_id"].nunique()),
    }
    out_json = args.out_dir / "probe_census.json"
    out_json.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")

    print(f"\n{'=' * 78}")
    print(f"  manifesto: {out_parquet}  ({len(manifest):,} linhas, "
          f"{manifest['variant_id'].nunique():,} variantes unicas)")
    print(f"  censo:     {out_json}")
    print(f"{'=' * 78}\n  Manda o probe_census.json de volta antes de rodar a GPU.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
