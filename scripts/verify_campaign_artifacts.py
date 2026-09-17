#!/usr/bin/env python3
"""Confere os artefatos da campanha ANTES de treinar: seleção comum x snapshots finais, com hashes.

Roda no notebook (pandas + pyarrow; sem GPU). So le; escreve um relatorio se --out-dir for dado.
Plano: docs/proposta_mosaic_regionalizacao_desenvolvimento.md, secao 4.2.

POR QUE EXISTE
--------------
O `final_para_treino` do relatorio do G2 diz apenas que UMA lista de clusters foi passada -- nao que foi a lista
certa, nem que o resultado ficou disjunto. Treinar nao pode depender desse booleano. Aqui a conferencia e feita
contra os artefatos de verdade:

1. nenhuma variante do conjunto de selecao aparece em QUALQUER papel de QUALQUER snapshot;
2. nenhum cluster do conjunto de selecao aparece no treino de nenhum snapshot;
3. nenhum membro dos estudos brasileiros aparece em nenhum snapshot (a garantia do G2, reconferida aqui);
4. validacao e teste sao identicos entre os candidatos -- se divergirem, os candidatos nao sao comparaveis;
5. os sha256 dos arquivos sao registrados, e conferidos contra --esperado quando fornecido.

Qualquer falha PARA com codigo 2: e o portao que o treino chama antes de comecar.

O QUE NAO PROVA
---------------
- Nao valida a politica de isolamento escolhida, que e decisao declarada fora daqui.
- Nao mede exposicao; para isso ha `measure_study_locus_exposure.py`.

USO (notebook)
--------------
    export WORK=~/testeArq/lumina-beat-regionalization
    PYTHONPATH="$WORK" python3 "$WORK"/scripts/verify_campaign_artifacts.py \
        --selection ~/artifacts/redesenho/g5_comum/selecao_comum.parquet \
        --brazil-variants ~/artifacts/redesenho/g1_brazil_studies/brazil_study_variants.parquet \
        --snapshot nenhum=~/artifacts/redesenho/g2_final_nenhum/core_head_snapshot.parquet \
        --snapshot janela2048=~/artifacts/redesenho/g2_final_janela2048/core_head_snapshot.parquet \
        --snapshot janela4096=~/artifacts/redesenho/g2_final_janela4096/core_head_snapshot.parquet
    echo "exit=$?"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_core_locus_head_snapshot import ROLE_TEST, ROLE_TRAIN, ROLE_VALIDATION  # noqa: E402
from scripts.import_mosaic_brazil_studies import sha256_file  # noqa: E402


def parse_snapshot(texto: str) -> tuple[str, Path]:
    if "=" not in texto:
        raise argparse.ArgumentTypeError(f"use nome=caminho, recebi {texto!r}")
    nome, caminho = texto.split("=", 1)
    return nome, Path(caminho).expanduser()


def role_signature(snapshot: pd.DataFrame, role: str) -> tuple[int, str]:
    """Assinatura de um papel: quantas linhas e a concatenacao ordenada dos ids."""
    ids = sorted(snapshot.loc[snapshot["role"] == role, "variant_id"].astype(str))
    return len(ids), "|".join(ids)


def check_snapshot(
    nome: str, snapshot: pd.DataFrame, *, selection: pd.DataFrame, members: set[str]
) -> tuple[list[str], dict[str, Any]]:
    problems: list[str] = []
    sel_ids = set(selection["variant_id"].astype(str))
    sel_clusters = set(selection["overlap_cluster_id"].dropna().astype(str))
    treino = snapshot[snapshot["role"] == ROLE_TRAIN]

    em_comum = sorted(set(snapshot["variant_id"].astype(str)) & sel_ids)
    if em_comum:
        problems.append(f"{nome}: {len(em_comum)} variantes do conjunto de selecao dentro do snapshot, "
                        f"ex.: {em_comum[:3]}")
    clusters = sorted(set(treino["overlap_cluster_id"].dropna().astype(str)) & sel_clusters)
    if clusters:
        problems.append(f"{nome}: {len(clusters)} clusters do conjunto de selecao no treino, ex.: {clusters[:3]}")
    membros = sorted(set(snapshot["variant_id"].astype(str)) & members)
    if membros:
        problems.append(f"{nome}: {len(membros)} membros dos estudos dentro do snapshot, ex.: {membros[:3]}")

    resumo = {
        "linhas": int(len(snapshot)),
        "treino": int(len(treino)),
        "treino_P": int((treino["binary_label"].astype("Int64") == 1).sum()),
        "clusters_treino": int(treino["overlap_cluster_id"].nunique()),
        "validacao": int((snapshot["role"] == ROLE_VALIDATION).sum()),
        "teste": int((snapshot["role"] == ROLE_TEST).sum()),
    }
    return problems, resumo


def check_comparability(assinaturas: dict[str, dict[str, tuple[int, str]]]) -> list[str]:
    """Os candidatos so sao comparaveis se validacao e teste forem exatamente os mesmos exemplos."""
    problems: list[str] = []
    for role in (ROLE_VALIDATION, ROLE_TEST):
        distintas = {nome: assinatura[role] for nome, assinatura in assinaturas.items()}
        referencia = next(iter(distintas.values()))
        divergentes = [nome for nome, valor in distintas.items() if valor != referencia]
        if divergentes:
            tamanhos = {nome: valor[0] for nome, valor in distintas.items()}
            problems.append(f"{role} difere entre candidatos ({divergentes}); tamanhos {tamanhos}")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--brazil-variants", required=True, type=Path)
    parser.add_argument("--snapshot", action="append", required=True, type=parse_snapshot,
                        metavar="NOME=CAMINHO", help="pode repetir; um por candidato")
    parser.add_argument("--esperado", type=Path,
                        help="JSON {arquivo: sha256} declarado antes; divergencia falha")
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args(argv)

    selection_path = args.selection.expanduser()
    selection = pd.read_parquet(selection_path,
                                columns=["variant_id", "overlap_cluster_id", "binary_label", "primary_panel"])
    members = set(pd.read_parquet(args.brazil_variants.expanduser(),
                                  columns=["variant_id"])["variant_id"].astype(str))

    problems: list[str] = []
    resumos: dict[str, Any] = {}
    assinaturas: dict[str, dict[str, tuple[int, str]]] = {}
    hashes: dict[str, str] = {str(selection_path): sha256_file(selection_path)}

    for nome, caminho in args.snapshot:
        snapshot = pd.read_parquet(caminho, columns=["variant_id", "role", "binary_label", "overlap_cluster_id"])
        p, resumo = check_snapshot(nome, snapshot, selection=selection, members=members)
        problems += p
        resumos[nome] = resumo
        assinaturas[nome] = {role: role_signature(snapshot, role) for role in (ROLE_VALIDATION, ROLE_TEST)}
        hashes[str(caminho)] = sha256_file(caminho)

    problems += check_comparability(assinaturas)

    if args.esperado is not None:
        esperado = json.loads(args.esperado.expanduser().read_text(encoding="utf-8"))
        for arquivo, digest in esperado.items():
            atual = hashes.get(arquivo)
            if atual is None:
                problems.append(f"{arquivo}: declarado em --esperado mas nao conferido nesta execucao")
            elif atual != digest:
                problems.append(f"{arquivo}: sha256 {atual} difere do declarado {digest}")

    relatorio = {
        "selecao": {"arquivo": str(selection_path), "variantes": int(len(selection)),
                    "clusters": int(selection["overlap_cluster_id"].nunique())},
        "candidatos": resumos,
        "sha256": hashes,
        "validacao_e_teste_identicos": not any("difere entre candidatos" in p for p in problems),
        "status": "FALHOU" if problems else "OK",
        "problemas": problems,
        "o_que_nao_prova": [
            "nao valida a politica de isolamento escolhida, que e declarada fora daqui",
            "nao mede exposicao de locus",
        ],
    }
    if args.out_dir is not None:
        out_dir = args.out_dir.expanduser()
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "verificacao_de_artefatos.json").write_text(
            json.dumps(relatorio, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps({k: relatorio[k] for k in ("selecao", "candidatos", "validacao_e_teste_identicos", "status")},
                     ensure_ascii=False, indent=2))
    if problems:
        print(f"\nFALHOU: {len(problems)} problema(s).")
        for problem in problems:
            print(f"  - {problem}")
        return 2
    print("\nsha256:")
    for arquivo, digest in hashes.items():
        print(f"  {digest[:16]}  {arquivo}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
