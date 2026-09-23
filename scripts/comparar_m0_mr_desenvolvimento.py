#!/usr/bin/env python3
"""Comparacao EXPLORATORIA M0 x MR no conjunto de selecao comum (desenvolvimento).

Treina H0 sobre o cache do M0 e HR sobre o cache do MR com a MESMA receita, as MESMAS sementes de cabeca (pareadas:
M0_i e MR_i usam h_i) e a extracao e a politica que o G5 escolheu SO COM M0. Mede AUROC, AUPRC e a macro dos paineis
no conjunto de selecao, por semente e na media das probabilidades calibradas, e o delta MR - M0 com IC por
bootstrap de `overlap_cluster_id`, com os mesmos sorteios para os dois sistemas.

O QUE ISTO NAO MEDE
    A interacao regional: os recortes de desenvolvimento excluem os membros dos estudos e a regra ampla brasileira,
    entao nao ha casos de participacao brasileira aqui. Isto e classificacao geral. Com um so adapter (a_1), a
    variacao entre sementes de adapter nao entra. Os IC sao exploratorios e nao sao criterio de avanco.
    E o conjunto de selecao ja escolheu a extracao e a politica do M0 (a melhor de 6 configuracoes): a macro do M0
    ali tende a estar sorteada para cima, o que, se tanto, puxa o delta MR - M0 para baixo.

O QUE O CODIGO EXIGE ANTES DE TREINAR
    - a decisao do G5, feita com o MESMO cache do M0 (sha256 da identidade);
    - M0 e MR com identidades iguais em tudo menos o adapter (tabela, codigo, ambiente, lote, janela, FASTA);
    - o adapter do MR igual ao declarado congelado para a semente.

USO (notebook)
    PYTHONPATH="$WORK" python3 scripts/comparar_m0_mr_desenvolvimento.py \\
        --cache-m0 ~/artifacts/redesenho/g3_cache/M0 --cache-mr ~/artifacts/redesenho/g3_cache/MR_a1 \\
        --decisao-g5 ~/artifacts/redesenho/g5/g5_decisao.json \\
        --snapshot nenhum=... --snapshot janela2048=... --snapshot janela4096=... \\
        --out-dir ~/artifacts/redesenho/comparacao_dev_a1
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

from eval.campanha import metricas  # noqa: E402
from eval.campanha.leitura_do_cache import carregar_cache, conferir_par, linhas_da_politica  # noqa: E402
from eval.campanha.recortes import adapter_congelado, carregar_campanha  # noqa: E402
from scripts.g5_escolher_extracao_e_politica import (  # noqa: E402
    hashes_dos_snapshots,
    ler_snapshots,
    sha_da_identidade,
)


def _delta(a: Any, b: Any) -> float | None:
    return None if a is None or b is None else float(b - a)


def comparar(rodadas_m0: list[dict], rodadas_mr: list[dict], tabela: pd.DataFrame, linhas: dict[str, np.ndarray],
             *, replicas: int, seed: int) -> dict[str, Any]:
    """Metricas por semente, da media das probabilidades e o bootstrap pareado da media. Funcao pura sobre as
    probabilidades ja calculadas, para ser testavel sem torch."""
    selecao = linhas["selecao"]
    y = tabela["binary_label"].astype(int).to_numpy()[selecao]
    paineis = tabela["primary_panel"].astype(str).to_numpy()[selecao]
    clusters = tabela["overlap_cluster_id"].astype(str).to_numpy()[selecao]
    por_semente = []
    for r0, rr in zip(rodadas_m0, rodadas_mr):
        if r0["semente"] != rr["semente"]:
            raise ValueError(f"sementes desalinhadas: {r0['semente']} x {rr['semente']}")
        por_semente.append({"semente": r0["semente"],
                            "M0": {k: r0["selecao"][k] for k in ("macro", "auroc", "auprc")},
                            "MR": {k: rr["selecao"][k] for k in ("macro", "auroc", "auprc")},
                            "delta": {k: _delta(r0["selecao"][k], rr["selecao"][k]) for k in ("macro", "auroc", "auprc")},
                            "epoca": {"M0": r0["epoca"], "MR": rr["epoca"]}})
    media_m0 = np.mean([r["prob_selecao"] for r in rodadas_m0], axis=0)
    media_mr = np.mean([r["prob_selecao"] for r in rodadas_mr], axis=0)
    resumo_m0 = metricas.resumo(media_m0, y, paineis)
    resumo_mr = metricas.resumo(media_mr, y, paineis)
    return {
        "por_semente": por_semente,
        "media_das_probabilidades": {"M0": resumo_m0, "MR": resumo_mr,
                                     "delta": {k: _delta(resumo_m0[k], resumo_mr[k])
                                               for k in ("macro", "auroc", "auprc")}},
        "bootstrap_da_media": metricas.bootstrap_pareado_por_cluster(media_m0, media_mr, y, paineis, clusters,
                                                                     replicas=replicas, seed=seed),
    }


def snapshots_diferentes(decisao_g5: dict[str, Any], atuais: dict[str, str]) -> list[str]:
    """Politicas cujo snapshot nao e o mesmo arquivo usado na decisao do G5. Troca silenciosa entre a escolha e a
    comparacao mudaria o treino da cabeca sem mudar nenhum nome (revisao de 23/09)."""
    registrados = decisao_g5.get("snapshots_sha256") or {}
    if not registrados:
        return ["a decisao do G5 nao registrou os hashes dos snapshots"]
    return sorted(p for p in set(registrados) | set(atuais) if registrados.get(p) != atuais.get(p))


def salvar_cabecas(destino: Path, sistema: str, rodadas: list[dict], contexto: dict[str, Any]) -> list[str]:
    """Cada cabeca por inteiro -- pesos, padronizacao, Platt e limiar --, para pontuar os estudos com o sistema
    efetivamente congelado, sem retreinar."""
    import torch

    from eval.campanha.cabeca import RECEITA

    caminhos = []
    for rodada in rodadas:
        caminho = destino / f"cabeca_{sistema}_h{rodada['semente']}.pt"
        torch.save({"formato": "campanha_r03_cabeca_v1", "sistema": sistema, "semente": rodada["semente"],
                    "estado": rodada["modelo"]["estado"], "media": rodada["modelo"]["media"],
                    "desvio": rodada["modelo"]["desvio"], "platt": rodada["platt"],
                    "limiar_de_mcc": rodada["limiar_de_mcc"], "epoca": rodada["epoca"], "receita": RECEITA,
                    **contexto}, caminho)
        caminhos.append(str(caminho))
    return caminhos


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache-m0", required=True, type=Path)
    parser.add_argument("--cache-mr", required=True, type=Path)
    parser.add_argument("--decisao-g5", required=True, type=Path)
    parser.add_argument("--snapshot", action="append", default=[], help="politica=caminho, uma vez por politica")
    parser.add_argument("--campanha", type=Path, default=Path("configs/campanha_r03_desenvolvimento.json"))
    parser.add_argument("--replicas", type=int, default=1000)
    parser.add_argument("--seed-do-bootstrap", type=int, default=20260901)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args(argv)

    campanha = carregar_campanha(args.campanha)
    decisao_g5 = json.loads(args.decisao_g5.expanduser().read_text(encoding="utf-8"))
    if decisao_g5.get("cache_m0_identidade_sha256") != sha_da_identidade(args.cache_m0):
        print("FALHOU: a decisao do G5 foi tomada com OUTRO cache do M0; rode o G5 com este cache antes")
        return 2
    extracao, politica = decisao_g5["decisao"]["extracao"], decisao_g5["decisao"]["politica"]
    snapshots = ler_snapshots(args.snapshot)
    trocados = snapshots_diferentes(decisao_g5, hashes_dos_snapshots(snapshots))
    if trocados:
        print(f"FALHOU: snapshots diferentes dos usados na decisao do G5: {trocados}")
        return 2
    m0, mr = carregar_cache(args.cache_m0, extracao), carregar_cache(args.cache_mr, extracao)
    conferir_par(m0, mr)
    semente_do_adapter = mr["identidade"]["semente_do_adapter"]
    if adapter_congelado(campanha, semente_do_adapter)["sha256"] != mr["identidade"]["adapter_sha256"]:
        print("FALHOU: o adapter do cache MR nao e o declarado congelado para a semente")
        return 2
    linhas = linhas_da_politica(m0, pd.read_parquet(snapshots[politica]))
    sementes = list(campanha["sementes"]["cabeca"])
    print(f"[comparacao] extracao {extracao} | politica {politica} | adapter {semente_do_adapter} | cabecas {sementes}")

    from eval.campanha.cabeca import RECEITA, rodar_sementes, sem_matrizes

    inicio = time.perf_counter()
    rodadas_m0 = rodar_sementes(m0["matriz"], m0["tabela"], linhas, sementes=sementes, device=args.device)
    rodadas_mr = rodar_sementes(mr["matriz"], mr["tabela"], linhas, sementes=sementes, device=args.device)
    resultado = comparar(rodadas_m0, rodadas_mr, m0["tabela"], linhas, replicas=args.replicas,
                         seed=args.seed_do_bootstrap)
    validacao = {"M0": metricas.resumo(np.mean([r["prob_validacao"] for r in rodadas_m0], axis=0),
                                       m0["tabela"]["binary_label"].astype(int).to_numpy()[linhas["validation"]],
                                       m0["tabela"]["primary_panel"].astype(str).to_numpy()[linhas["validation"]]),
                 "MR": metricas.resumo(np.mean([r["prob_validacao"] for r in rodadas_mr], axis=0),
                                       mr["tabela"]["binary_label"].astype(int).to_numpy()[linhas["validation"]],
                                       mr["tabela"]["primary_panel"].astype(str).to_numpy()[linhas["validation"]]),
                 "leitura": "so descritivo: as cabecas pararam e foram calibradas neste fold"}
    relatorio = {
        "natureza": "exploratoria, desenvolvimento",
        "extracao": extracao, "politica": politica, "semente_do_adapter": semente_do_adapter,
        "sementes_da_cabeca": sementes, "receita_da_cabeca": RECEITA,
        "treino": int(len(linhas["train"])), "validacao": int(len(linhas["validation"])),
        "selecao": int(len(linhas["selecao"])),
        **resultado,
        "fold1_descritivo": validacao,
        "cabecas": {"M0": [sem_matrizes(r) for r in rodadas_m0], "MR": [sem_matrizes(r) for r in rodadas_mr]},
        "identidades": {"M0": str(args.cache_m0), "MR": str(args.cache_mr),
                        "decisao_g5": str(args.decisao_g5)},
        "o_que_nao_mede": ("a interacao regional (sem casos de participacao brasileira nos recortes de "
                           "desenvolvimento); a variacao entre sementes de adapter (so a_1)"),
        "vies_conhecido": ("o conjunto de selecao escolheu a configuracao do M0 no G5 (a melhor de 6): a macro do M0 "
                           "tende a estar sorteada para cima e o delta MR - M0, se tanto, puxado para baixo"),
        "segundos": round(time.perf_counter() - inicio, 1),
    }
    destino = args.out_dir.expanduser()
    destino.mkdir(parents=True, exist_ok=True)
    contexto = {"extracao": extracao, "politica": politica,
                "snapshot_sha256": hashes_dos_snapshots(snapshots)[politica],
                "decisao_g5_sha256": hashlib.sha256(args.decisao_g5.expanduser().read_bytes()).hexdigest()}
    relatorio["cabecas_salvas"] = {
        "M0": salvar_cabecas(destino, "M0", rodadas_m0, {**contexto, "cache_identidade_sha256":
                                                         sha_da_identidade(args.cache_m0)}),
        "MR": salvar_cabecas(destino, "MR", rodadas_mr, {**contexto, "cache_identidade_sha256":
                                                         sha_da_identidade(args.cache_mr),
                                                         "semente_do_adapter": semente_do_adapter})}
    (destino / "comparacao_m0_mr.json").write_text(json.dumps(relatorio, ensure_ascii=False, indent=2, default=str),
                                                   encoding="utf-8")
    ids = m0["tabela"]["variant_id"].to_numpy()[linhas["selecao"]]
    pd.DataFrame({"variant_id": ids,
                  **{f"M0_h{r['semente']}": r["prob_selecao"] for r in rodadas_m0},
                  **{f"MR_h{r['semente']}": r["prob_selecao"] for r in rodadas_mr}}).to_parquet(
        destino / "predicoes_selecao.parquet", index=False)

    media = resultado["media_das_probabilidades"]
    boot = resultado["bootstrap_da_media"]
    print("\n== conjunto de selecao, media das 3 cabecas (EXPLORATORIO) ==")
    for chave in ("macro", "auroc", "auprc"):
        faixa = boot[chave]
        print(f"  {chave:<6} M0 {media['M0'][chave]:.4f}  MR {media['MR'][chave]:.4f}  delta "
              f"{media['delta'][chave]:+.4f}  IC [{faixa['p2_5']:+.4f}; {faixa['p97_5']:+.4f}] "
              f"({faixa['replicas_validas']} replicas)")
    print("  por painel (AUROC):")
    for painel in metricas.PAINEIS_DE_DISCRIMINACAO + metricas.PAINEIS_DE_GUARDA:
        a, b = media["M0"]["por_painel"][painel], media["MR"]["por_painel"][painel]
        valor = lambda x: "-" if x is None else f"{x:.4f}"  # noqa: E731
        print(f"    {painel:<11} M0 {valor(a['auroc'])}  MR {valor(b['auroc'])}  (P {a['n_pos']}, B {a['n_neg']})")
    print("  por semente (macro):")
    for linha in resultado["por_semente"]:
        print(f"    h{linha['semente']}: M0 {linha['M0']['macro']:.4f}  MR {linha['MR']['macro']:.4f}  "
              f"delta {linha['delta']['macro']:+.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
