#!/usr/bin/env python3
"""G6: confere o sistema que vai ao G7, fixa o limiar do ensemble no fold 1 e monta o manifesto.

Nada e treinado; o fold 0 e os estudos nao sao lidos.

O QUE FAZ
    1. le a composicao final declarada (`g6.composicao_final`) e recusa qualquer desvio do pareamento de sementes;
    2. cada componente tem de ser o arquivo que a conferencia do seu comparador aprovou (mesmo sha256), e o M0 dos
       comparadores da a_2 e da a_3 tem de ter sido conferido identico ao da a_1 -- aqui ele e reconferido;
    3. confere os quatro caches (M0, MR_a1, MR_a2, MR_a3): sistema, adapter congelado da semente, o mesmo R03 da
       referencia, M0 x MR so diferindo pelo adapter, as mesmas linhas de fold 1 e de selecao;
    4. confere as LINHAS por IDs e hashes: cada cabeca aponta (cache_identidade_sha256) para a identidade do seu
       cache, que fixa o hash de conteudo da tabela, papel incluido (reconferido ao carregar); os IDs do fold 1 sao
       exatamente o papel `validation` do snapshot da politica, e os da selecao exatamente os de
       `selecao_comum.parquet` (sha256 com o prefixo declarado), nas contagens declaradas;
    5. recarrega as seis cabecas e confere metadados (decisao do G5, snapshot, cache, adapter, receita, Platt a > 0)
       e reproducao: probabilidades da selecao contra `predicoes_selecao.parquet`; metricas da selecao e do fold 1
       contra o relatorio do comparador; Platt e limiar da cabeca refeitos no fold 1. Isso e consistencia NUMERICA;
       a identidade das linhas vem do passo 4, nao de os valores coincidirem;
    6. predicao do sistema = media das tres probabilidades calibradas; limiar do ensemble = a regra do Mosaic
       (`calibrate_threshold`) nessa media, no fold 1, por sistema;
    7. compara o ensemble FINAL no conjunto de selecao, com IC por cluster. DESCRITIVO: nao muda composicao nem
       limiar, e os ICs sao condicionais aos sistemas treinados;
    8. monta o manifesto. Com bloqueio, grava so o RASCUNHO; com --congelar e sem bloqueio, grava tambem
       `g6_manifesto.json` e o sha256 dos bytes a parte. Os bloqueios conferem CONTEUDO, nao o texto do estado:
       margens com estudos, delta, metrica, estatistica e limite finito; unidade e configuracao do bootstrap;
       pendencia FEITO com `onde` e RETIRADO com `motivo`; revisao do git conferida e codigo sem mudanca fora do
       commit (um git que falha bloqueia, nao vale como `sem mudancas`); ABraOM reconferido no arquivo.

SAIDAS (em --out-dir, que nao pode existir)
    g6_construcao.json           conferencias, reproducoes, limiares, ensemble final no desenvolvimento, bloqueios
    g6_predicoes.parquet         fold 1 e selecao: probabilidade de cada componente e de cada sistema
    g6_manifesto_rascunho.json   sempre que as conferencias passam
    g6_manifesto.json (+ .sha256) so com --congelar e sem bloqueio
Se uma conferencia falhar, so `g6_construcao.json` e gravado (saida 2).

USO (notebook)
    PYTHONPATH="$WORK" python3 scripts/construir_g6.py --raiz ~/artifacts/redesenho \\
        --cache-m0 ~/artifacts/redesenho/g3_cache/M0 \\
        --cache-mr 20260921=~/artifacts/redesenho/g3_cache/MR_a1 \\
        --cache-mr 20260922=~/artifacts/redesenho/g3_cache/MR_a2 \\
        --cache-mr 20260923=~/artifacts/redesenho/g3_cache/MR_a3 \\
        --decisao-g5 ~/artifacts/redesenho/g5/g5_decisao.json \\
        --snapshot-da-politica ~/artifacts/redesenho/g2_final_janela2048/core_head_snapshot.parquet \\
        --selecao ~/artifacts/redesenho/g5_comum/selecao_comum.parquet \\
        --proveniencia abraom=~/artifacts/redesenho/g0_fontes/SABE1171.Abraom.clean.tsv \\
        --out-dir ~/artifacts/redesenho/g6_rascunho
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from eval.campanha import g6, metricas  # noqa: E402
from eval.campanha.cache import sha256_do_arquivo  # noqa: E402
from eval.campanha.leitura_do_cache import carregar_cache, diferencas_de_identidade  # noqa: E402
from eval.campanha.recortes import adapter_congelado, carregar_campanha  # noqa: E402
from scripts.g5_escolher_extracao_e_politica import sha_da_identidade  # noqa: E402

#: A do conferidor das cabecas: recarregar no mesmo ambiente reproduz exatamente; a tolerancia so evita reprovar
#: por ordem de soma noutra CPU.
TOLERANCIA = 1e-6
PAPEIS = ("validation", "selecao")


def ler_pares(pares: list[str], opcao: str) -> dict[str, Path]:
    saida = {}
    for par in pares:
        chave, sinal, caminho = par.partition("=")
        if not sinal or not chave.strip() or not caminho.strip():
            raise SystemExit(f"FALHOU: {opcao} espera chave=caminho, recebeu {par!r}")
        if chave.strip() in saida:
            raise SystemExit(f"FALHOU: {opcao} {chave.strip()} repetido")
        saida[chave.strip()] = Path(caminho.strip()).expanduser()
    return saida


def _git(*argumentos: str) -> str:
    """Saida do git; codigo de saida diferente de zero LEVANTA (check=True): falha nunca vira saida vazia."""
    return subprocess.run(["git", *argumentos], cwd=RAIZ, capture_output=True, text=True, timeout=30,
                          check=True).stdout


def estado_do_codigo(arquivos: list[str]) -> dict[str, Any]:
    """Revisao do git, arquivos ausentes, fora do git (nao rastreados) e com mudanca fora do commit. Se o git falhar,
    `erro` diz por que -- e o congelamento bloqueia (g6.problemas_do_codigo)."""
    ausentes = [a for a in arquivos if not (RAIZ / a).exists()]
    presentes = [a for a in arquivos if a not in ausentes]
    try:
        revisao = _git("rev-parse", "HEAD").strip()
        rastreados = set(_git("ls-files", "--", *presentes).split())
        status = _git("status", "--porcelain", "--", *presentes)
    except (OSError, subprocess.SubprocessError) as exc:
        detalhe = getattr(exc, "stderr", None) or exc
        return {"revisao": None, "ausentes": ausentes, "nao_rastreados": [], "modificados": [],
                "erro": f"{type(exc).__name__}: {str(detalhe).strip()[:300]}"}
    nao_rastreados = sorted(a for a in presentes if a not in rastreados)
    modificados = sorted({linha[3:].strip() for linha in status.splitlines() if linha.strip()} - set(nao_rastreados))
    return {"revisao": revisao, "ausentes": ausentes, "nao_rastreados": nao_rastreados, "modificados": modificados,
            "erro": None}


def conferir_proveniencia(campanha: dict[str, Any], arquivos: dict[str, Path]) -> tuple[dict[str, Any], list[str]]:
    """sha256 dos arquivos de proveniencia passados, contra o declarado (sha256 completo ou prefixo)."""
    declarada = campanha["g6"]["proveniencia"]
    dados = declarada["dados_do_adapter"]
    referencia = campanha["adapter_do_mr"]["referencia"]
    esperados = {"abraom": declarada["abraom"]["sha256"],
                 "pool_abraom": dados["pool_abraom"]["sha256_prefixo"],
                 "pool_global": dados["pool_global"]["sha256_prefixo"],
                 "plano": dados["plano"]["sha256_prefixo"],
                 "plano_treino": referencia["plano_treino_sha256"],
                 "plano_validacao": referencia["plano_validacao_sha256"]}
    saida, problemas = {}, []
    for nome, caminho in sorted(arquivos.items()):
        if nome not in esperados:
            problemas.append(f"--proveniencia {nome}: fora de {sorted(esperados)}")
            continue
        if not caminho.exists():
            problemas.append(f"--proveniencia {nome}: {caminho} nao existe")
            continue
        sha, esperado = sha256_do_arquivo(caminho), esperados[nome]
        confere = sha == esperado if len(esperado) == 64 else sha.startswith(esperado)
        saida[nome] = {"arquivo": str(caminho), "sha256": sha, "declarado": esperado, "confere": confere}
        if not confere:
            problemas.append(f"--proveniencia {nome}: sha256 {sha[:12]} nao confere com o declarado {esperado[:12]}")
    return saida, problemas


def conferir_linhas(ids: dict[str, np.ndarray], *, snapshot: Path, selecao: Path,
                    campanha: dict[str, Any]) -> list[str]:
    """A identidade das linhas por IDs e hashes: o fold 1 do cache e exatamente o papel `validation` do snapshot da
    politica e a selecao e exatamente `selecao_comum.parquet`, nas contagens declaradas. Do snapshot so se usam os
    IDs do papel `validation`: o fold 0 (papel `test`) nao entra."""
    problemas = []
    recortes = campanha["recortes"]
    tabela = pd.read_parquet(snapshot, columns=["variant_id", "role"])
    do_snapshot = set(tabela.loc[tabela["role"] == "validation", "variant_id"].astype(str))
    if len(ids["validation"]) != len(set(ids["validation"])) or set(ids["validation"]) != do_snapshot:
        problemas.append(f"fold 1 do cache ({len(ids['validation'])}) nao e o papel validation do snapshot da politica "
                         f"({len(do_snapshot)}; {len(set(ids['validation']) ^ do_snapshot)} ids diferem)")
    esperado = recortes["parada_e_calibracao"].get("variantes")
    if len(ids["validation"]) != esperado:
        problemas.append(f"fold 1 com {len(ids['validation'])} variantes; a declaracao diz {esperado}")
    prefixo = recortes["comparacao_de_desenvolvimento"]["sha256_prefixo"]
    sha = sha256_do_arquivo(selecao)
    if not sha.startswith(prefixo):
        problemas.append(f"--selecao com sha256 {sha[:12]}, a declaracao diz {prefixo}")
    comum = set(pd.read_parquet(selecao, columns=["variant_id"])["variant_id"].astype(str))
    if len(ids["selecao"]) != len(set(ids["selecao"])) or set(ids["selecao"]) != comum:
        problemas.append(f"selecao do cache ({len(ids['selecao'])}) nao e selecao_comum.parquet ({len(comum)}; "
                         f"{len(set(ids['selecao']) ^ comum)} ids diferem)")
    return problemas


def ambiente_da_pontuacao() -> dict[str, Any]:
    import platform

    import torch

    return {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "torch": torch.__version__, "dispositivo": "cpu", "threads": torch.get_num_threads()}


def _falhar(problemas: list[str]) -> int:
    print(f"\nFALHOU: {len(problemas)} problema(s)")
    for problema in problemas:
        print(f"  - {problema}")
    return 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raiz", required=True, type=Path, help="pasta dos comparadores (a de composicao_final)")
    parser.add_argument("--cache-m0", required=True, type=Path)
    parser.add_argument("--cache-mr", action="append", default=[], help="semente_do_adapter=pasta, uma por adapter")
    parser.add_argument("--decisao-g5", required=True, type=Path)
    parser.add_argument("--snapshot-da-politica", required=True, type=Path,
                        help="o core_head_snapshot.parquet da politica escolhida no G5")
    parser.add_argument("--selecao", required=True, type=Path, help="g5_comum/selecao_comum.parquet")
    parser.add_argument("--entrada", action="append", default=[],
                        help="nome=arquivo de entrada das analises secundarias (regra_ampla, exposicao)")
    parser.add_argument("--proveniencia", action="append", default=[],
                        help="nome=arquivo para reconferir (abraom, pool_abraom, pool_global, plano, plano_treino, "
                             "plano_validacao)")
    parser.add_argument("--campanha", type=Path, default=RAIZ / "configs" / "campanha_r03_desenvolvimento.json")
    parser.add_argument("--superficie", type=Path, default=RAIZ / "configs" / "adapter_r03_superficie.json")
    parser.add_argument("--replicas", type=int, default=1000)
    parser.add_argument("--seed-do-bootstrap", type=int, default=20260901)
    parser.add_argument("--congelar", action="store_true",
                        help="grava g6_manifesto.json (+ .sha256) se nao houver bloqueio; senao, saida 2")
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args(argv)

    inicio = time.perf_counter()
    destino = args.out_dir.expanduser()
    if destino.exists():
        print(f"FALHOU: {destino} ja existe; o G6 grava sempre numa pasta nova")
        return 2
    campanha = carregar_campanha(args.campanha)
    try:
        componentes = g6.componentes_declarados(campanha)
    except ValueError as exc:
        return _falhar([str(exc)])
    raiz = args.raiz.expanduser()
    problemas: list[str] = []

    # --- decisao do G5, snapshot, comparadores e conferencias: tudo barato, antes de carregar caches
    caminho_da_decisao = args.decisao_g5.expanduser()
    decisao = json.loads(caminho_da_decisao.read_text(encoding="utf-8"))
    extracao, politica = decisao["decisao"]["extracao"], decisao["decisao"]["politica"]
    sha_da_decisao = sha256_do_arquivo(caminho_da_decisao)
    sha_do_snapshot = sha256_do_arquivo(args.snapshot_da_politica.expanduser())
    if sha_do_snapshot != (decisao.get("snapshots_sha256") or {}).get(politica):
        problemas.append(f"--snapshot-da-politica ({sha_do_snapshot[:12]}) nao e o snapshot {politica} registrado na "
                         f"decisao do G5")
    nomes = sorted({c["comparador"] for s in g6.SISTEMAS for c in componentes[s]})
    relatorios, conferencias, predicoes, sha_das_conferencias = {}, {}, {}, {}
    for nome in nomes:
        pasta = raiz / nome
        faltando = [a for a in ("comparacao_m0_mr.json", "conferencia_das_cabecas.json", "predicoes_selecao.parquet")
                    if not (pasta / a).exists()]
        if faltando:
            problemas.append(f"{pasta}: sem {faltando}")
            continue
        relatorios[nome] = json.loads((pasta / "comparacao_m0_mr.json").read_text(encoding="utf-8"))
        conferencias[nome] = json.loads((pasta / "conferencia_das_cabecas.json").read_text(encoding="utf-8"))
        sha_das_conferencias[nome] = sha256_do_arquivo(pasta / "conferencia_das_cabecas.json")
        predicoes[nome] = pd.read_parquet(pasta / "predicoes_selecao.parquet")
        if (relatorios[nome].get("extracao"), relatorios[nome].get("politica")) != (extracao, politica):
            problemas.append(f"{nome}: comparador com {relatorios[nome].get('extracao')}/"
                             f"{relatorios[nome].get('politica')}, a decisao do G5 e {extracao}/{politica}")
    sha_dos_arquivos = {}
    for c in (c for s in g6.SISTEMAS for c in componentes[s]):
        if (raiz / c["arquivo"]).exists():
            sha_dos_arquivos[c["arquivo"]] = sha256_do_arquivo(raiz / c["arquivo"])
        else:
            problemas.append(f"{raiz / c['arquivo']} nao existe")
    problemas += g6.conferir_conferencias(conferencias, componentes, sha_dos_arquivos)

    pastas = {"M0": args.cache_m0.expanduser(), **ler_pares(args.cache_mr, "--cache-mr")}
    esperadas = {"M0"} | {str(c["adapter"]) for c in componentes["MR"]}
    if set(pastas) != esperadas:
        problemas.append(f"caches {sorted(pastas)}, esperados {sorted(esperadas)} (--cache-mr semente=pasta)")
    identidades, sha_das_identidades = {}, {}
    for chave, pasta in pastas.items():
        if not (pasta / "identidade.json").exists():
            problemas.append(f"{pasta}: sem identidade.json")
            continue
        identidades[chave] = json.loads((pasta / "identidade.json").read_text(encoding="utf-8"))
        sha_das_identidades[chave] = sha_da_identidade(pasta)
    checkpoint = campanha["adapter_do_mr"]["referencia"]["checkpoint_sha256"]
    for chave, identidade in identidades.items():
        if identidade.get("checkpoint_sha256") != checkpoint:
            problemas.append(f"cache {chave}: R03 {str(identidade.get('checkpoint_sha256'))[:12]} != a referencia "
                             f"{checkpoint[:12]}")
        if chave == "M0":
            if identidade.get("sistema") != "M0" or identidade.get("adapter_sha256") is not None:
                problemas.append("--cache-m0 nao e um cache M0 sem adapter")
            continue
        if identidade.get("sistema") != "MR" or identidade.get("semente_do_adapter") != int(chave):
            problemas.append(f"--cache-mr {chave}: cache {identidade.get('sistema')} da semente "
                             f"{identidade.get('semente_do_adapter')}")
        if identidade.get("adapter_sha256") != adapter_congelado(campanha, int(chave))["sha256"]:
            problemas.append(f"cache MR {chave}: adapter {str(identidade.get('adapter_sha256'))[:12]} nao e o "
                             f"congelado para a semente")
        if "M0" in identidades:
            diferentes = diferencas_de_identidade(identidades["M0"], identidade)
            if diferentes:
                problemas.append(f"cache MR {chave} difere do M0 alem do adapter, em {diferentes}")
    if problemas:
        return _falhar(problemas)

    proveniencia, problemas_de_proveniencia = conferir_proveniencia(
        campanha, ler_pares(args.proveniencia, "--proveniencia"))
    entradas = {}
    for nome, caminho in ler_pares(args.entrada, "--entrada").items():
        if nome not in g6.ENTRADAS_DAS_ANALISES_SECUNDARIAS:
            problemas_de_proveniencia.append(f"--entrada {nome}: fora de {sorted(g6.ENTRADAS_DAS_ANALISES_SECUNDARIAS)}")
        elif not caminho.exists():
            problemas_de_proveniencia.append(f"--entrada {nome}: {caminho} nao existe")
        else:
            entradas[nome] = {"arquivo": str(caminho), "sha256": sha256_do_arquivo(caminho),
                              "descricao": g6.ENTRADAS_DAS_ANALISES_SECUNDARIAS[nome]}
    if problemas_de_proveniencia:
        return _falhar(problemas_de_proveniencia)

    # --- caches: completos e com a tabela da identidade; guarda so as linhas de fold 1 e de selecao
    print(f"[g6] extracao {extracao} | politica {politica} | carregando {len(pastas)} caches")
    matrizes: dict[str, dict[str, np.ndarray]] = {}
    tabela_ref: pd.DataFrame | None = None
    ids: dict[str, np.ndarray] = {}
    for chave in ["M0", *sorted(set(pastas) - {"M0"})]:
        cache = carregar_cache(pastas[chave], extracao)
        tabela = cache["tabela"]
        papel = tabela["papel"].astype(str).to_numpy()
        indices = {p: np.nonzero(papel == p)[0] for p in PAPEIS}
        if tabela_ref is None:
            tabela_ref = pd.concat([tabela.iloc[indices[p]] for p in PAPEIS], ignore_index=True)
            ids = {p: tabela["variant_id"].astype(str).to_numpy()[indices[p]] for p in PAPEIS}
        elif any(not np.array_equal(tabela["variant_id"].astype(str).to_numpy()[indices[p]], ids[p]) for p in PAPEIS):
            problemas.append(f"cache {chave}: linhas de fold 1 ou de selecao diferentes das do M0")
        matrizes[chave] = {p: cache["matriz"][indices[p]] for p in PAPEIS}
        del cache
        print(f"  {chave:<9} ok ({', '.join(f'{p} {len(indices[p])}' for p in PAPEIS)})")
    assert tabela_ref is not None
    rotulos = {p: tabela_ref.loc[tabela_ref["papel"] == p, "binary_label"].astype(int).to_numpy() for p in PAPEIS}
    paineis = {p: tabela_ref.loc[tabela_ref["papel"] == p, "primary_panel"].astype(str).to_numpy() for p in PAPEIS}
    clusters = tabela_ref.loc[tabela_ref["papel"] == "selecao", "overlap_cluster_id"].astype(str).to_numpy()
    declarado = campanha["recortes"]["comparacao_de_desenvolvimento"]
    if (len(ids["selecao"]), len(set(clusters))) != (declarado["variantes"], declarado["clusters"]):
        problemas.append(f"selecao com {len(ids['selecao'])} variantes em {len(set(clusters))} clusters; a declaracao "
                         f"diz {declarado['variantes']} em {declarado['clusters']}")
    problemas += conferir_linhas(ids, snapshot=args.snapshot_da_politica.expanduser(),
                                 selecao=args.selecao.expanduser(), campanha=campanha)
    for nome, tabela_pred in predicoes.items():
        if not np.array_equal(tabela_pred["variant_id"].astype(str).to_numpy(), ids["selecao"]):
            problemas.append(f"{nome}/predicoes_selecao.parquet nao esta na ordem das linhas de selecao do cache")
    if problemas:
        return _falhar(problemas)

    # --- as seis cabecas: metadados e reproducao
    from eval.campanha.cabeca import carregar_cabeca_salva, limiar_de_mcc, platt, pontuar_salva
    from scripts.conferir_cabecas_salvas import conferir_metadados, diferencas_de_cabeca

    registros, probabilidades = [], {s: {p: [] for p in PAPEIS} for s in g6.SISTEMAS}
    por_componente: dict[str, dict[str, np.ndarray]] = {}
    for sistema in g6.SISTEMAS:
        for c in componentes[sistema]:
            chave = "M0" if sistema == "M0" else str(c["adapter"])
            relatorio = relatorios[c["comparador"]]
            cabeca = carregar_cabeca_salva(raiz / c["arquivo"])
            esperado = {"decisao_g5_sha256": sha_da_decisao, "snapshot_sha256": sha_do_snapshot,
                        "cache_identidade_sha256": {"M0": sha_das_identidades["M0"],
                                                    "MR": sha_das_identidades.get(chave)}}
            proprios = conferir_metadados(cabeca, sistema=sistema, semente=c["cabeca"], relatorio=relatorio,
                                          esperado=esperado)
            if sistema == "MR" and relatorio.get("semente_do_adapter") != c["adapter"]:
                proprios.append(f"o comparador {c['comparador']} e da semente de adapter "
                                f"{relatorio.get('semente_do_adapter')}")
            logits, prob = {}, {}
            for p in PAPEIS:
                logits[p], prob[p] = pontuar_salva(cabeca, matrizes[chave][p])
                probabilidades[sistema][p].append(prob[p])
            por_componente[c["rotulo"]] = prob
            coluna = predicoes[c["comparador"]].get(c["coluna"])
            dp_selecao = (float(np.max(np.abs(prob["selecao"] - coluna.to_numpy()))) if coluna is not None
                          else float("inf"))
            if not dp_selecao <= TOLERANCIA:
                proprios.append(f"nao reproduz {c['coluna']} de predicoes_selecao.parquet: max|dp| {dp_selecao:.3e}")
            por_semente = next((x for x in relatorio["por_semente"] if x["semente"] == c["cabeca"]), {})
            proprios += g6.diferencas_numericas(metricas.resumo(logits["selecao"], rotulos["selecao"],
                                                                paineis["selecao"]),
                                                por_semente.get(sistema, {}), ("macro", "auroc", "auprc"),
                                                tolerancia=TOLERANCIA, rotulo="selecao (logits)")
            no_relatorio = next((x for x in relatorio["cabecas"][sistema] if x["semente"] == c["cabeca"]), {})
            proprios += g6.diferencas_numericas(metricas.resumo(logits["validation"], rotulos["validation"],
                                                                paineis["validation"]),
                                                no_relatorio.get("validacao", {}), ("macro", "auroc", "auprc"),
                                                tolerancia=TOLERANCIA, rotulo="fold 1 (logits)")
            a, b = platt(logits["validation"], rotulos["validation"])
            proprios += g6.diferencas_numericas({"a": a, "b": b}, cabeca["platt"], ("a", "b"),
                                                tolerancia=TOLERANCIA, rotulo="Platt refeito no fold 1")
            proprios += g6.diferencas_numericas(limiar_de_mcc(prob["validation"], rotulos["validation"]),
                                                cabeca["limiar_de_mcc"], ("limiar", "mcc"), tolerancia=TOLERANCIA,
                                                rotulo="limiar da cabeca refeito no fold 1")
            registros.append({
                "sistema": sistema, "rotulo": c["rotulo"], "cabeca": c["cabeca"], "adapter": c["adapter"],
                "arquivo": c["arquivo"], "sha256": sha_dos_arquivos[c["arquivo"]], "epoca": cabeca["epoca"],
                "platt": {"a": cabeca["platt"]["a"], "b": cabeca["platt"]["b"]},
                "limiar_da_cabeca": cabeca["limiar_de_mcc"],
                "cache": {"chave": chave, "identidade_sha256": sha_das_identidades[chave]},
                "reproducao": {"max_dp_selecao": dp_selecao, "platt_refeito": {"a": a, "b": b}},
                "problemas": proprios})
            problemas += [f"{c['arquivo']}: {p}" for p in proprios]

    referencia_m0 = componentes["M0"][0]["comparador"]
    m0_nos_comparadores: dict[str, Any] = {}
    for nome in nomes:
        if nome == referencia_m0:
            continue
        m0_nos_comparadores[nome] = {}
        for c in componentes["M0"]:
            outra = raiz / nome / Path(c["arquivo"]).name
            diferencas = (diferencas_de_cabeca(carregar_cabeca_salva(outra), carregar_cabeca_salva(raiz / c["arquivo"]))
                          if outra.exists() else ["ausente"])
            m0_nos_comparadores[nome][Path(c["arquivo"]).name] = diferencas or "identica"
            problemas += [f"{outra}: difere de {c['arquivo']}: {d}" for d in diferencas]

    # --- ensemble, limiar e o ensemble final no desenvolvimento
    media = {s: {p: g6.media_dos_componentes(probabilidades[s][p]) for p in PAPEIS} for s in g6.SISTEMAS}
    limiares = {s: g6.limiar_do_mosaic(rotulos["validation"], media[s]["validation"]) for s in g6.SISTEMAS}
    fold1_descritivo = {s: metricas.resumo(media[s]["validation"], rotulos["validation"], paineis["validation"])
                        for s in g6.SISTEMAS}
    do_m0 = relatorios[referencia_m0]
    problemas += g6.diferencas_numericas(
        metricas.resumo(media["M0"]["selecao"], rotulos["selecao"], paineis["selecao"]),
        do_m0["media_das_probabilidades"]["M0"], ("macro", "auroc", "auprc"), tolerancia=TOLERANCIA,
        rotulo=f"ensemble M0 na selecao contra {referencia_m0}")
    problemas += g6.diferencas_numericas(fold1_descritivo["M0"], do_m0["fold1_descritivo"]["M0"],
                                         ("macro", "auroc", "auprc"), tolerancia=TOLERANCIA,
                                         rotulo=f"ensemble M0 no fold 1 contra {referencia_m0}")
    desenvolvimento = g6.comparacao_do_ensemble(media["M0"]["selecao"], media["MR"]["selecao"], rotulos["selecao"],
                                                paineis["selecao"], clusters, limiares=limiares,
                                                replicas=args.replicas, seed=args.seed_do_bootstrap)

    arquivos_de_codigo = sorted(set(g6.CODIGO_DO_G6) | set(g6.CODIGO_DO_G7))
    codigo = {**estado_do_codigo(arquivos_de_codigo),
              "arquivos": {a: sha256_do_arquivo(RAIZ / a) for a in arquivos_de_codigo if (RAIZ / a).exists()}}
    bloqueios = g6.bloqueios(campanha, estado_do_codigo=codigo, proveniencia=proveniencia, entradas=entradas)
    fold1 = {"papel": "validation", "descricao": "fold 1 gold (validation_gold do run 0)",
             "variantes": int(len(ids["validation"])), "n_P": int(rotulos["validation"].sum()),
             "n_B": int((rotulos["validation"] == 0).sum()), "ids_sha256": g6.sha256_dos_ids(ids["validation"]),
             "igual_ao_papel_validation_do_snapshot": True, "snapshot_sha256": sha_do_snapshot}
    selecao = {"variantes": int(len(ids["selecao"])), "clusters": int(len(set(clusters))),
               "ids_sha256": g6.sha256_dos_ids(ids["selecao"]), "igual_a_selecao_comum": True,
               "selecao_comum_sha256": sha256_do_arquivo(args.selecao.expanduser())}
    caches = {chave: {"pasta": str(pastas[chave]), "identidade_sha256": sha_das_identidades[chave],
                      **{k: identidades[chave].get(k) for k in ("sistema", "semente_do_adapter", "adapter_sha256",
                                                                "checkpoint_sha256", "tabela_sha256_conteudo",
                                                                "versao_do_extrator", "janela_bp", "fasta_sha256",
                                                                "revisao_do_codigo")}}
              for chave in sorted(pastas)}
    extracao_da_campanha = {"extracao": extracao, "codigo": identidades["M0"].get("codigo"),
                            "ambiente": identidades["M0"].get("ambiente"), "lote": identidades["M0"].get("lote"),
                            "regra": "os caches dos estudos tem de ter a identidade do cache de desenvolvimento do "
                                     "mesmo sistema em tudo menos a tabela"}
    construcao = {
        "formato": "campanha_r03_g6_construcao_v1", "passou": not problemas, "problemas": problemas,
        "bloqueios": bloqueios, "extracao": extracao, "politica": politica,
        "componentes": registros, "m0_nos_comparadores": m0_nos_comparadores, "caches": caches,
        "fold1": fold1, "selecao": selecao, "limiares_do_ensemble": limiares,
        "fold1_descritivo": {**fold1_descritivo, "leitura": "so descritivo: as cabecas pararam, foram calibradas e o "
                                                            "limiar do ensemble foi escolhido neste fold"},
        "desenvolvimento_do_ensemble_final": {**desenvolvimento, "como_ler": g6.COMO_LER_O_DESENVOLVIMENTO},
        "proveniencia_reconferida": proveniencia, "codigo": codigo,
        "segundos": round(time.perf_counter() - inicio, 1),
    }
    destino.mkdir(parents=True)
    (destino / "g6_construcao.json").write_text(json.dumps(construcao, ensure_ascii=False, indent=2, default=str),
                                                 encoding="utf-8")
    if problemas:
        return _falhar(problemas)

    colunas = {"variant_id": np.concatenate([ids[p] for p in PAPEIS]),
               "papel": np.concatenate([[p] * len(ids[p]) for p in PAPEIS])}
    for rotulo, prob in por_componente.items():
        colunas[rotulo] = np.concatenate([prob[p] for p in PAPEIS])
    for sistema in g6.SISTEMAS:
        colunas[f"ensemble_{sistema}"] = np.concatenate([media[sistema][p] for p in PAPEIS])
    pd.DataFrame(colunas).to_parquet(destino / "g6_predicoes.parquet", index=False)

    superficie = json.loads(args.superficie.expanduser().read_text(encoding="utf-8"))
    comum = dict(
        declaracao_sha256=sha256_do_arquivo(args.campanha.expanduser()),
        decisao_g5={"arquivo": str(caminho_da_decisao), "sha256": sha_da_decisao, "extracao": extracao,
                    "politica": politica, "snapshot_sha256": sha_do_snapshot,
                    "snapshot_arquivo": str(args.snapshot_da_politica.expanduser())},
        componentes=registros, caches=caches, extracao=extracao_da_campanha, limiares=limiares, fold1=fold1,
        conferencias={nome: {"sha256": sha_das_conferencias[nome], "passou": conferencias[nome]["passou"],
                             "m0_de_referencia": (conferencias[nome].get("m0_de_referencia") or {}).get("cabecas")}
                      for nome in nomes},
        proveniencia=proveniencia, codigo=codigo, ambiente_da_pontuacao=ambiente_da_pontuacao(),
        entradas=entradas, raiz_dos_comparadores=str(raiz),
        modulos=len(superficie), bloqueios_atuais=bloqueios,
        criado_em_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
    (destino / g6.NOME_DO_RASCUNHO).write_bytes(g6.serializar(g6.montar_manifesto(campanha, **comum)))

    _imprimir(construcao, desenvolvimento, limiares, destino)
    if bloqueios:
        print(f"\n== manifesto: RASCUNHO ({destino / g6.NOME_DO_RASCUNHO}); {len(bloqueios)} bloqueio(s) ==")
        for bloqueio in bloqueios:
            print(f"  - {bloqueio}")
        if args.congelar:
            print("FALHOU: --congelar com bloqueio; nada foi congelado")
            return 2
        print("As conferencias passaram. O congelamento espera os bloqueios acima.")
        return 0
    if not args.congelar:
        print(f"\n== manifesto: RASCUNHO sem bloqueio ({destino / g6.NOME_DO_RASCUNHO}); congele com --congelar ==")
        return 0
    sha = g6.gravar_manifesto(destino, g6.montar_manifesto(campanha, **comum, estado=g6.CONGELADO))
    g6.ler_manifesto_congelado(destino)
    print(f"\n== manifesto CONGELADO: {destino / g6.NOME_DO_MANIFESTO} (sha256 {sha}) ==")
    return 0


def _imprimir(construcao: dict[str, Any], desenvolvimento: dict[str, Any], limiares: dict[str, Any],
              destino: Path) -> None:
    valor = lambda x: "-" if x is None else f"{x:.4f}"  # noqa: E731
    fold1, selecao = construcao["fold1"], construcao["selecao"]
    print(f"\n[g6] {construcao['extracao']} | {construcao['politica']} | fold 1: {fold1['variantes']} "
          f"(P {fold1['n_P']}, B {fold1['n_B']}) | selecao: {selecao['variantes']} em {selecao['clusters']} clusters")
    print("  componente            sha256        epoca  platt a   platt b   max|dp| selecao  reproducao")
    for r in construcao["componentes"]:
        print(f"  {r['rotulo']:<20}  {r['sha256'][:12]}  {r['epoca']:>5}  {r['platt']['a']:>8.4f}  "
              f"{r['platt']['b']:>8.4f}  {r['reproducao']['max_dp_selecao']:.1e}          "
              f"{'ok' if not r['problemas'] else 'PROBLEMA'}")
    for nome, cabecas in construcao["m0_nos_comparadores"].items():
        iguais = sum(1 for v in cabecas.values() if v == "identica")
        print(f"  M0 em {nome}: {iguais} de {len(cabecas)} cabecas identicas as da composicao")
    print("\n== limiar do ensemble: regra do Mosaic na media das probabilidades, no fold 1 ==")
    for sistema, l in limiares.items():
        print(f"  {sistema}: limiar {l['threshold']:.6f}  MCC {l['mcc']:.4f}  especificidade {l['specificity']:.4f}  "
              f"sensibilidade {l['sensitivity']:.4f}")
    print("\n== desenvolvimento: ENSEMBLE FINAL no conjunto de selecao (EXPLORATORIO; nao muda composicao nem "
          "limiar) ==")
    boot = desenvolvimento["bootstrap"]
    for chave in ("macro", "auroc", "auprc"):
        faixa = boot[chave]
        print(f"  {chave:<6} M0 {valor(desenvolvimento['M0'][chave])}  MR {valor(desenvolvimento['MR'][chave])}  "
              f"delta {desenvolvimento['delta'][chave]:+.4f}  IC [{faixa['p2_5']:+.4f}; {faixa['p97_5']:+.4f}] "
              f"({faixa['replicas_validas']} replicas)")
    print("  por painel (AUROC):")
    for painel in metricas.PAINEIS_DE_DISCRIMINACAO + metricas.PAINEIS_DE_GUARDA:
        a, b = desenvolvimento["M0"]["por_painel"][painel], desenvolvimento["MR"]["por_painel"][painel]
        fragil = "  fragil: < 10 de uma classe" if a["auroc"] is not None and min(a["n_pos"], a["n_neg"]) < 10 else ""
        print(f"    {painel:<11} M0 {valor(a['auroc'])}  MR {valor(b['auroc'])}  (P {a['n_pos']}, B {a['n_neg']})"
              f"{fragil}")
    for sistema, m in desenvolvimento["com_o_limiar_do_fold1"].items():
        print(f"  com o limiar do fold 1, {sistema}: MCC {valor(m['mcc'])}  sensibilidade {valor(m['sensitivity'])}  "
              f"especificidade {valor(m['specificity'])}")
    print("  ICs condicionais aos sistemas treinados (reamostram clusters, nao o treino); detalhe em "
          f"{destino / 'g6_construcao.json'}")


if __name__ == "__main__":
    sys.exit(main())
