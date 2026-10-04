#!/usr/bin/env python3
"""Fase 1, passo 1: inventario do Mosaic v1 novo (`mosaic-v1-2026-09-30`) para o R03 congelado.

So le: nao treina, nao extrai, nao avalia. Mostra o que a Fase 1 pede ao release e quanto os caches antigos do M0
ja cobrem (docs/fase1_r03_congelado.md):

- papeis por execucao na vista de 4 kb (treino sem purgas, validation gold sem purgas, teste gold elegivel e
  teste de todos os tiers), para conferir com o guia de splits (run 0: 194.666 / 2.106 / 2.110);
- os dois proxies brasileiros (papeis, P/B, sem par por painel, presenca no ABraOM de casos e controles);
- a coorte de beneficio (gold presente no ABraOM) e as P-BR (P/LP presentes no ABraOM, por tier, com os grupos de
  gene da vista);
- as 13 variantes criticas, uma a uma, inclusive as que nao estao no release ou no ABraOM;
- os pedidos do manifesto (`outputs/candidate-manifest/`), por estudo, papel e execucao nula;
- a cobertura dos caches antigos sobre as variantes elegiveis do release e sobre os pedidos, e o complemento.

USO (notebook)
    PYTHONPATH="$PWD" python3 scripts/inventario_mosaic_v1.py --entrega ~/mosaic-v1-2026-09-30 \\
        --cache-antigo ~/artifacts/redesenho/g3_cache/M0 --cache-antigo ~/artifacts/redesenho/g7_cache/M0 \\
        --out-dir ~/artifacts/mosaic_v1/inventario

SAIDAS (em --out-dir, que nao pode existir): inventario.json e complemento_release_4kb.txt (ids elegiveis em 4 kb
que nenhum cache antigo cobre).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

RELEASE = "artifacts/mosaic-v1-2026-09-30"
VISTA = "views/4kb/partitions.parquet"
MEMBERSHIP = "studies/brazilian-proxies/membership.parquet"
PEDIDOS = "outputs/candidate-manifest"
CRITICAS = "config/critical-variants-br.yaml"
K = 5
CAMPOS_DA_IDENTIDADE = ("versao_do_extrator", "sistema", "checkpoint_sha256", "adapter_sha256", "fasta_sha256",
                        "janela_bp", "indice_focal", "lote", "extracoes", "revisao_do_codigo")


def sha256(caminho: Path) -> str:
    digest = hashlib.sha256()
    with open(caminho, "rb") as arquivo:
        for bloco in iter(lambda: arquivo.read(1 << 20), b""):
            digest.update(bloco)
    return digest.hexdigest()


def carregar_release(raiz: Path) -> pd.DataFrame:
    """Uma linha por variante do release: rotulo, coordenadas, vista de 4 kb, painel e as anotacoes usadas aqui."""
    exemplos = pd.read_parquet(raiz / "clinical-variants.parquet",
                               columns=["variant_id", "chrom", "pos_1based", "ref", "alt", "binary_label",
                                        "label_tier", "br_lab_any"])
    vista = pd.read_parquet(raiz / VISTA, columns=["variant_id", "sequence_eligible", "overlap_cluster_id",
                                                   "core_unit_id", "core_fold", "core_purged_runs",
                                                   "gene_transfer_group_id"])
    paineis = pd.read_parquet(raiz / "evaluation-panels.parquet", columns=["variant_id", "primary_panel"])
    anotacoes = pd.read_parquet(raiz / "variant-annotations.parquet",
                                columns=["variant_id", "present_abraom", "abraom_status", "abraom_filter",
                                         "gnomad_status"])
    df = (exemplos.merge(vista, on="variant_id", validate="one_to_one")
          .merge(paineis, on="variant_id", validate="one_to_one")
          .merge(anotacoes, on="variant_id", validate="one_to_one"))
    if len(df) != len(exemplos):
        raise ValueError(f"o release perdeu linhas no join: {len(df)} de {len(exemplos)}")
    df["present_abraom"] = df["present_abraom"].fillna(False).astype(bool)
    df["sequence_eligible"] = df["sequence_eligible"].astype(bool)
    return df


def purgada(df: pd.DataFrame, run: int) -> np.ndarray:
    return np.array([run in (runs if runs is not None else ()) for runs in df["core_purged_runs"]], dtype=bool)


def papeis(df: pd.DataFrame, run: int) -> dict[str, np.ndarray]:
    """As mascaras da execucao `run` no `core_locus` de 4 kb, como o guia de splits monta."""
    teste, validacao = run, (run + 1) % K
    fold = df["core_fold"].to_numpy()
    sem_purga = ~purgada(df, run)
    gold = (df["label_tier"] == "gold").to_numpy()
    elegivel = df["sequence_eligible"].to_numpy()
    treino = ~np.isin(fold, [teste, validacao]) & sem_purga & df["label_tier"].isin(["gold", "consensus"]).to_numpy()
    return {"treino": treino, "validation": (fold == validacao) & sem_purga & gold,
            "teste_gold": (fold == teste) & gold & elegivel, "teste_todos": (fold == teste) & elegivel}


def contagens_por_execucao(df: pd.DataFrame) -> list[dict[str, int]]:
    saida = []
    for run in range(K):
        mascaras = papeis(df, run)
        saida.append({"run": run, **{nome: int(m.sum()) for nome, m in mascaras.items()},
                      "purgadas": int(purgada(df, run).sum())})
    return saida


def _pb(linhas: pd.DataFrame) -> dict[str, int]:
    return {"n": int(len(linhas)), "n_P": int((linhas["binary_label"] == 1).sum()),
            "n_B": int((linhas["binary_label"] == 0).sum())}


def proxies(membership: pd.DataFrame, df: pd.DataFrame) -> dict[str, Any]:
    presenca = df.set_index("variant_id")["present_abraom"]
    saida: dict[str, Any] = {}
    for estudo, linhas in membership.groupby("study_id", sort=True):
        bloco: dict[str, Any] = {papel: _pb(g) for papel, g in linhas.groupby("member_role", sort=True)}
        sem_par = linhas[linhas["member_role"] == "unmatched_case"]
        bloco["sem_par_por_painel"] = {painel: _pb(g) for painel, g in sem_par.groupby("primary_panel", sort=True)}
        bloco["presente_no_abraom"] = {
            papel: float(presenca.reindex(g["variant_id"]).fillna(False).astype(bool).mean())
            for papel, g in linhas.groupby("member_role", sort=True)}
        saida[str(estudo)] = bloco
    return saida


def beneficio_e_pbr(df: pd.DataFrame) -> dict[str, Any]:
    """Coorte de beneficio (teste gold elegivel presente no ABraOM, somando as cinco execucoes) e P-BR."""
    elegivel = df["sequence_eligible"]
    beneficio = df[elegivel & (df["label_tier"] == "gold") & df["present_abraom"]]
    pbr = df[elegivel & (df["binary_label"] == 1) & df["present_abraom"]]
    return {"beneficio": {**_pb(beneficio), "unidades": int(beneficio["overlap_cluster_id"].nunique())},
            "p_br": {"por_tier": {tier: int(n) for tier, n in pbr["label_tier"].value_counts().sort_index().items()},
                     "grupos_de_gene": int(pbr["gene_transfer_group_id"].nunique()), "n": int(len(pbr))}}


def ler_criticas(caminho: Path) -> list[dict[str, Any]]:
    import yaml

    return list(yaml.safe_load(caminho.read_text(encoding="utf-8"))["variants"])


def criticas(lista: list[dict[str, Any]], df: pd.DataFrame) -> list[dict[str, Any]]:
    """Cada critica pela coordenada, esteja ou nao no release ou no ABraOM (a lacuna do evaluate_safety.py)."""
    chave = df["chrom"] + ":" + df["pos_1based"].astype(str) + ":" + df["ref"] + ":" + df["alt"]
    por_chave = df.assign(_chave=chave.to_numpy()).set_index("_chave")
    saida = []
    for item in lista:
        g = item["grch38"]
        alvo = f"{g['chrom']}:{g['pos']}:{g['ref']}:{g['alt']}"
        registro: dict[str, Any] = {"gene": item.get("gene"), "hgvs": item.get("hgvs"), "chave": alvo,
                                    "no_release": alvo in por_chave.index}
        if registro["no_release"]:
            linha = por_chave.loc[alvo]
            if isinstance(linha, pd.DataFrame):
                linha = linha.iloc[0]
            registro.update({"variant_id": linha["variant_id"], "tier": linha["label_tier"],
                             "binary_label": int(linha["binary_label"]),
                             "present_abraom": bool(linha["present_abraom"]),
                             "elegivel_4kb": bool(linha["sequence_eligible"]), "core_fold": int(linha["core_fold"])})
        saida.append(registro)
    return saida


def pedidos(pasta: Path) -> dict[str, Any]:
    requests = pd.read_parquet(pasta / "requests.parquet", columns=["variant_id", "study", "run", "role"])
    variants = pd.read_parquet(pasta / "variants.parquet", columns=["variant_id", "in_release"])
    por_estudo = {}
    for estudo, g in requests.groupby("study", sort=True):
        por_estudo[str(estudo)] = {"linhas": int(len(g)), "variantes": int(g["variant_id"].nunique()),
                                   "run_nulo": int(g["run"].isna().sum()),
                                   "papeis": {str(p): int(n) for p, n in g["role"].value_counts().sort_index().items()}}
    return {"por_estudo": por_estudo, "variantes": int(len(variants)),
            "variantes_no_release": int(variants["in_release"].fillna(False).astype(bool).sum())}


def ids_do_cache(pasta: Path) -> set[str]:
    ids: set[str] = set()
    for fragmento in sorted(pasta.glob("fragmento_*.npz")):
        with open(fragmento, "rb") as arquivo, np.load(arquivo, allow_pickle=False) as dados:
            ids.update(dados["variant_id"].astype(str).tolist())
    return ids


def cobertura_dos_caches(caches: list[Path], alvos: dict[str, set[str]]) -> tuple[dict[str, Any], set[str]]:
    """Identidade resumida de cada cache, quanto ele cobre de cada alvo e a uniao dos ids."""
    saida: dict[str, Any] = {}
    uniao: set[str] = set()
    for pasta in caches:
        identidade_arquivo = pasta / "identidade.json"
        identidade = json.loads(identidade_arquivo.read_text(encoding="utf-8")) if identidade_arquivo.exists() else {}
        ids = ids_do_cache(pasta)
        uniao |= ids
        saida[str(pasta)] = {"identidade": {c: identidade.get(c) for c in CAMPOS_DA_IDENTIDADE},
                             "variantes": len(ids),
                             "cobre": {nome: len(ids & alvo) for nome, alvo in alvos.items()}}
    return saida, uniao


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--entrega", required=True, type=Path, help="checkout dedicado do Mosaic com a entrega baixada")
    parser.add_argument("--cache-antigo", action="append", default=[], type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    entrega = args.entrega.expanduser()
    destino = args.out_dir.expanduser()
    if destino.exists():
        print(f"FALHOU: {destino} ja existe; o inventario grava sempre numa pasta nova")
        return 2
    raiz = entrega / RELEASE
    df = carregar_release(raiz)
    membership = pd.read_parquet(raiz / MEMBERSHIP)
    elegiveis = set(df.loc[df["sequence_eligible"], "variant_id"])
    alvos = {"release_elegivel_4kb": elegiveis}
    pasta_pedidos = entrega / PEDIDOS
    resumo_pedidos = None
    if (pasta_pedidos / "requests.parquet").exists():
        resumo_pedidos = pedidos(pasta_pedidos)
        requests = pd.read_parquet(pasta_pedidos / "requests.parquet", columns=["variant_id", "study"])
        for estudo, g in requests.groupby("study", sort=True):
            alvos[f"pedidos:{estudo}"] = set(g["variant_id"].astype(str))
    caches, uniao = cobertura_dos_caches([p.expanduser() for p in args.cache_antigo], alvos)
    complemento = sorted(elegiveis - uniao)
    relatorio = {
        "formato": "inventario_mosaic_v1_fase1_v1",
        "entrega": str(entrega),
        "arquivos": {nome: sha256(raiz / nome) for nome in ("clinical-variants.parquet", VISTA,
                                                            "evaluation-panels.parquet", "variant-annotations.parquet",
                                                            MEMBERSHIP)},
        "release": {"variantes": int(len(df)), "elegiveis_4kb": len(elegiveis),
                    "por_tier": {t: int(n) for t, n in df["label_tier"].value_counts().sort_index().items()}},
        "execucoes_4kb": contagens_por_execucao(df),
        "proxies": proxies(membership, df),
        "regional_no_release": beneficio_e_pbr(df),
        "criticas": criticas(ler_criticas(entrega / CRITICAS), df),
        "pedidos": resumo_pedidos,
        "caches_antigos": caches,
        "complemento": {"release_elegivel_4kb": len(complemento),
                        **{nome: len(alvo - uniao) for nome, alvo in alvos.items() if nome != "release_elegivel_4kb"}},
    }
    destino.mkdir(parents=True)
    (destino / "inventario.json").write_text(json.dumps(relatorio, ensure_ascii=False, indent=2, default=str),
                                             encoding="utf-8")
    (destino / "complemento_release_4kb.txt").write_text("".join(f"{v}\n" for v in complemento), encoding="utf-8")
    imprimir(relatorio)
    print(f"\nINVENTARIO: {destino / 'inventario.json'}")
    return 0


def imprimir(r: dict[str, Any]) -> None:
    print(f"release: {r['release']['variantes']:,} variantes, {r['release']['elegiveis_4kb']:,} elegiveis em 4 kb, "
          f"tiers {r['release']['por_tier']}")
    for e in r["execucoes_4kb"]:
        print(f"  run {e['run']}: treino {e['treino']:,} | validation {e['validation']:,} | teste gold "
              f"{e['teste_gold']:,} | teste todos os tiers {e['teste_todos']:,} | purgadas {e['purgadas']:,}")
    for estudo, bloco in r["proxies"].items():
        papeis_ = {p: v for p, v in bloco.items() if p in ("case", "unmatched_case", "control")}
        print(f"{estudo}: {papeis_} | presente no ABraOM {bloco['presente_no_abraom']}")
        print(f"  sem par por painel: {bloco['sem_par_por_painel']}")
    reg = r["regional_no_release"]
    print(f"beneficio (gold presente no ABraOM): {reg['beneficio']} | P-BR {reg['p_br']}")
    for c in r["criticas"]:
        estado = (f"tier {c['tier']}, P/B {c['binary_label']}, ABraOM {c['present_abraom']}, elegivel "
                  f"{c['elegivel_4kb']}, fold {c['core_fold']}") if c["no_release"] else "FORA DO RELEASE"
        print(f"  critica {c['gene']} {c['chave']}: {estado}")
    if r["pedidos"]:
        print(f"pedidos: {r['pedidos']['variantes']:,} variantes ({r['pedidos']['variantes_no_release']:,} no release)")
        for estudo, b in r["pedidos"]["por_estudo"].items():
            print(f"  {estudo}: {b['linhas']:,} linhas, {b['variantes']:,} variantes, run nulo {b['run_nulo']:,}, "
                  f"papeis {b['papeis']}")
    for pasta, c in r["caches_antigos"].items():
        print(f"cache {pasta}: {c['variantes']:,} variantes; cobre {c['cobre']}")
        print(f"  identidade: {c['identidade']}")
    print(f"complemento a extrair: {r['complemento']}")


if __name__ == "__main__":
    sys.exit(main())
