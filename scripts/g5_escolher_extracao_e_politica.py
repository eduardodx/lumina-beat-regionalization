#!/usr/bin/env python3
"""G5: escolhe a extracao e a politica de isolamento SO COM M0, no conjunto de selecao comum.

Para cada extracao (2) e politica (3), treina uma cabeca por semente declarada (11, 12, 13) sobre o cache do M0 --
treino = o `train` da politica, parada e calibracao no fold 1 -- e mede a macro-AUROC no conjunto de selecao. A
escolha aplica as regras de `eval/campanha/g5.py`: a da politica fixada no plano (secao 4.2) e a da extracao
proposta em 23/09 e registrada na declaracao ANTES de qualquer score. Esta ultima exige
`--confirmo-a-regra-da-extracao`: quem roda confirma que leu a regra que vai decidir.

O MR nao entra aqui: a escolha nao pode depender do adapter.

SAIDAS (em --out-dir)
    g5_decisao.json    -- a escolha, as medias, cada semente, as regras aplicadas e a identidade do cache do M0
                          (o comparador M0 x MR confere que usa o MESMO cache)
    g5_predicoes.parquet -- probabilidades calibradas no conjunto de selecao, por extracao, politica e semente

USO (notebook)
    PYTHONPATH="$WORK" python3 scripts/g5_escolher_extracao_e_politica.py --confirmo-a-regra-da-extracao \\
        --cache-m0 ~/artifacts/redesenho/g3_cache/M0 \\
        --snapshot nenhum=~/artifacts/redesenho/g2_final_nenhum/core_head_snapshot.parquet \\
        --snapshot janela2048=~/artifacts/redesenho/g2_final_janela2048/core_head_snapshot.parquet \\
        --snapshot janela4096=~/artifacts/redesenho/g2_final_janela4096/core_head_snapshot.parquet \\
        --out-dir ~/artifacts/redesenho/g5
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from eval.campanha import g5  # noqa: E402
from eval.campanha.cache import sha256_do_arquivo  # noqa: E402
from eval.campanha.layout import EXTRACOES  # noqa: E402
from eval.campanha.leitura_do_cache import carregar_cache, linhas_da_politica  # noqa: E402
from eval.campanha.recortes import carregar_campanha  # noqa: E402


def sha_da_identidade(pasta: Path) -> str:
    return hashlib.sha256((Path(pasta).expanduser() / "identidade.json").read_bytes()).hexdigest()


def ler_snapshots(pares: list[str]) -> dict[str, Path]:
    snapshots = {}
    for par in pares:
        nome, _, caminho = par.partition("=")
        snapshots[nome.strip()] = Path(caminho.strip()).expanduser()
    faltando = sorted(set(g5.ISOLAMENTO) - set(snapshots))
    sobrando = sorted(set(snapshots) - set(g5.ISOLAMENTO))
    if faltando or sobrando:
        raise SystemExit(f"FALHOU: --snapshot precisa exatamente de {list(g5.ISOLAMENTO)} "
                         f"(faltando {faltando}, sobrando {sobrando})")
    return snapshots


def hashes_dos_snapshots(snapshots: dict[str, Path]) -> dict[str, str]:
    """sha256 de cada snapshot de politica: o comparador confere que recebeu os MESMOS arquivos da decisao."""
    return {politica: sha256_do_arquivo(caminho) for politica, caminho in sorted(snapshots.items())}


def media_ou_none(valores: list[float | None]) -> float | None:
    return None if any(v is None for v in valores) else float(np.mean(valores))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache-m0", required=True, type=Path)
    parser.add_argument("--snapshot", action="append", default=[], help="politica=caminho, uma vez por politica")
    parser.add_argument("--campanha", type=Path, default=Path("configs/campanha_r03_desenvolvimento.json"))
    parser.add_argument("--confirmo-a-regra-da-extracao", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args(argv)

    campanha = carregar_campanha(args.campanha)
    regras = campanha.get("g5") or {}
    print("[regras do G5]")
    print(f"  politica: {regras.get('regra_da_politica')}")
    print(f"  extracao: {regras.get('regra_da_extracao', {}).get('texto')}")
    if not args.confirmo_a_regra_da_extracao:
        print("FALHOU: a regra da extracao foi proposta em 23/09 e decide o resultado. Leia-a acima e rode de novo "
              "com --confirmo-a-regra-da-extracao")
        return 2
    snapshots = ler_snapshots(args.snapshot)
    sementes = list(campanha["sementes"]["cabeca"])
    from eval.campanha.cabeca import RECEITA, rodar_sementes, sem_matrizes

    inicio = time.perf_counter()
    medias: dict[tuple[str, str], float | None] = {}
    detalhe: list[dict[str, Any]] = []
    predicoes: list[pd.DataFrame] = []
    for extracao in EXTRACOES:
        cache = carregar_cache(args.cache_m0, extracao)
        if cache["identidade"].get("sistema") != "M0":
            print(f"FALHOU: o G5 so usa M0; o cache em {args.cache_m0} e {cache['identidade'].get('sistema')}")
            return 2
        for politica in g5.ISOLAMENTO:
            linhas = linhas_da_politica(cache, pd.read_parquet(snapshots[politica]))
            rodadas = rodar_sementes(cache["matriz"], cache["tabela"], linhas, sementes=sementes, device=args.device)
            macros = [r["selecao"]["macro"] for r in rodadas]
            medias[(extracao, politica)] = media_ou_none(macros)
            print(f"  {extracao:<20} {politica:<11} treino {len(linhas['train']):>7,}  macro na selecao "
                  f"{' / '.join('-' if m is None else f'{m:.4f}' for m in macros)}  media "
                  f"{'-' if medias[(extracao, politica)] is None else f'{medias[(extracao, politica)]:.4f}'}")
            for r in rodadas:
                detalhe.append({"extracao": extracao, "politica": politica, "treino": int(len(linhas["train"])),
                                **sem_matrizes(r)})
                predicoes.append(pd.DataFrame({
                    "extracao": extracao, "politica": politica, "semente": r["semente"],
                    "variant_id": cache["tabela"]["variant_id"].to_numpy()[linhas["selecao"]],
                    "prob": r["prob_selecao"]}))

    decisao = g5.escolha(medias)
    relatorio = {
        "decisao": decisao,
        "medias": {f"{e}|{p}": m for (e, p), m in medias.items()},
        "detalhe": detalhe,
        "regras": regras,
        "regra_da_extracao_confirmada_na_execucao": True,
        "sementes_da_cabeca": sementes,
        "receita_da_cabeca": RECEITA,
        "cache_m0": str(args.cache_m0),
        "cache_m0_identidade_sha256": sha_da_identidade(args.cache_m0),
        "snapshots": {p: str(c) for p, c in snapshots.items()},
        "snapshots_sha256": hashes_dos_snapshots(snapshots),
        "device": args.device,
        "segundos": round(time.perf_counter() - inicio, 1),
        "o_que_nao_e": "nao usa MR; nao mede o efeito regional; o conjunto de selecao e de desenvolvimento",
    }
    destino = args.out_dir.expanduser()
    destino.mkdir(parents=True, exist_ok=True)
    (destino / "g5_decisao.json").write_text(json.dumps(relatorio, ensure_ascii=False, indent=2, default=str),
                                              encoding="utf-8")
    pd.concat(predicoes, ignore_index=True).to_parquet(destino / "g5_predicoes.parquet", index=False)
    print(f"\n[G5] extracao = {decisao['extracao']} | politica = {decisao['politica']} | macro media "
          f"{decisao['macro_media']:.4f}")
    for extracao, info in decisao["por_extracao"].items():
        print(f"  {extracao:<20} politica pela regra: {info['politica']:<11} macro media {info['macro_media']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
