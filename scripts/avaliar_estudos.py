#!/usr/bin/env python3
"""G7, passo 2: pontua os dois estudos brasileiros com o sistema CONGELADO e mede, uma vez. Com --ensaio-sintetico,
exercita o MESMO caminho na membership real com scores SINTETICOS, antes do congelamento.

MODO REAL (exige o G6 congelado)
    1. manifesto: sha256 a parte e estado CONGELADO; a declaracao e o codigo do G6/G7 sao os dele (sha256 de cada
       arquivo) e estao commitados;
    2. release: membership e anotacoes com o hash logico do Mosaic igual a referencia declarada;
    3. entradas das analises secundarias (regra ampla, exposicao): sha256 igual ao do manifesto;
    4. caches dos estudos (um por sistema): identidade igual a do cache de desenvolvimento do MESMO sistema em tudo
       menos a tabela; conteudo da tabela = o da tabela OFICIAL (membership + pb_examples, hash logico conferido):
       ids, coordenadas e alelos amarrados ao release; cache completo;
    5. as seis cabecas (sha256 do manifesto) recarregadas e aplicadas ao cache do seu sistema; predicao do sistema =
       media das tres probabilidades calibradas; limiares = os do manifesto;
    6. consumidor por estudo, com Brier (so dos sistemas: probabilidades); baselines com regra, papel e cobertura
       explicitos (sem Brier: sao scores de ordenacao); e as margens declaradas, aplicadas mecanicamente. Replicas,
       seed e unidade principal da interacao sao as do MANIFESTO: --replicas ou --seed diferentes sao recusados.
MODO ENSAIO (--ensaio-sintetico; sem manifesto)
    passos 2, 3 e 6 com scores SINTETICOS para os sistemas e para as baselines: nenhum modelo e lido e nenhuma
    metrica real sai. Os scores reais das baselines so sao calculados para contar a cobertura. O relatorio sai
    marcado ENSAIO.

SAIDAS (em --out-dir, que nao pode existir): g7_relatorio.json (ensaio_relatorio.json no ensaio) e, no modo real,
g7_pontos.parquet (probabilidade de cada sistema e score de cada baseline, por variante).

USO (notebook)
    ensaio: PYTHONPATH="$PWD" python3 scripts/avaliar_estudos.py --ensaio-sintetico --replicas 50 \\
                --release-root ~/mosaic-v1 --mosaic-root ~/testeArq/lumina-mosaic \\
                --entrada regra_ampla=... --entrada exposicao=... --out-dir ~/artifacts/redesenho/g7_ensaio
    real:   ... --manifesto ~/artifacts/redesenho/g6_final --raiz ~/artifacts/redesenho \\
                --cache-estudos M0=... --cache-estudos 20260921=... --cache-estudos 20260922=... \\
                --cache-estudos 20260923=... --out-dir ~/artifacts/redesenho/g7
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from eval.campanha import baselines, estudos, g6, g7  # noqa: E402
from eval.campanha.cache import sha256_do_arquivo  # noqa: E402
from eval.campanha.leitura_do_cache import carregar_cache  # noqa: E402
from eval.campanha.recortes import carregar_campanha, hash_do_conteudo  # noqa: E402
from scripts import conferir_cobertura_das_baselines as cobertura  # noqa: E402
from scripts.construir_g6 import estado_do_codigo, ler_pares  # noqa: E402

AVISO_DO_ENSAIO = "ENSAIO: scores SINTETICOS; nenhum numero aqui e resultado de modelo ou de baseline"


def ler_release(raiz: Path, logical_contract: Any, referencia: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame,
                                                                                         pd.DataFrame, list[str]]:
    """Membership, anotacoes e a tabela oficial dos estudos (membership + pb_examples), com o hash logico dos tres
    parquets conferido contra a referencia declarada."""
    problemas = cobertura.conferir_contratos(raiz, logical_contract, referencia, tuple(cobertura.CHAVES_DO_RELEASE))
    membros = pd.read_parquet(raiz / cobertura.MEMBERSHIP)
    anotacoes = pd.read_parquet(raiz / cobertura.ANOTACOES, columns=list(baselines.COLUNAS_DAS_ANOTACOES))
    exemplos = pd.read_parquet(raiz / cobertura.EXEMPLOS, columns=["variant_id", *g7.COLUNAS_DE_SEQUENCIA,
                                                                    "binary_label", "label_tier"])
    return membros, anotacoes, g7.tabela_oficial(membros, exemplos), problemas


def ler_entradas(caminhos: dict[str, Path], registradas: dict[str, Any] | None) -> tuple[set[str], pd.DataFrame,
                                                                                         dict[str, Any], list[str]]:
    """A lista da regra ampla e a exposicao por membro; com o manifesto, o sha256 de cada uma tem de ser o dele."""
    problemas, hashes = [], {}
    for nome in g6.ENTRADAS_DAS_ANALISES_SECUNDARIAS:
        if nome not in caminhos or not caminhos[nome].exists():
            problemas.append(f"--entrada {nome} ausente ou inexistente")
            continue
        hashes[nome] = {"arquivo": str(caminhos[nome]), "sha256": sha256_do_arquivo(caminhos[nome])}
        if registradas is not None and (registradas.get(nome) or {}).get("sha256") != hashes[nome]["sha256"]:
            problemas.append(f"--entrada {nome}: sha256 {hashes[nome]['sha256'][:12]} != o do manifesto")
    if problemas:
        return set(), pd.DataFrame(), hashes, problemas
    regra_ampla = {linha.strip() for linha in caminhos["regra_ampla"].read_text(encoding="utf-8").splitlines()
                   if linha.strip()}
    exposicao = pd.read_parquet(caminhos["exposicao"])
    return regra_ampla, exposicao, hashes, []


def exposicao_do_estudo(exposicao: pd.DataFrame, estudo: str) -> pd.Series:
    linhas = exposicao[exposicao["study_id"] == estudo]
    return pd.Series(linhas["n_janela"].to_numpy(dtype=float), index=linhas["variant_id"].astype(str).to_numpy())


def pontuar_sistemas(manifesto: dict[str, Any], raiz: Path, pastas: dict[str, Path],
                     oficial: pd.DataFrame) -> tuple[dict[str, pd.Series], dict[str, Any], list[str]]:
    """As seis cabecas do manifesto sobre os caches dos estudos: media das tres probabilidades por sistema."""
    from eval.campanha.cabeca import carregar_cabeca_salva, pontuar_salva

    problemas, registro, carregados = [], {}, {}
    extracao = manifesto["extracao_dos_caches"]["extracao"]
    conteudo_oficial = hash_do_conteudo(oficial)
    for chave, pasta in sorted(pastas.items()):
        desenvolvimento = manifesto["caches_de_desenvolvimento"].get(chave)
        if desenvolvimento is None:
            problemas.append(f"--cache-estudos {chave}: fora dos sistemas do manifesto")
            continue
        arquivo = Path(desenvolvimento["pasta"]).expanduser() / "identidade.json"
        if not arquivo.exists() or sha256_do_arquivo(arquivo) != desenvolvimento["identidade_sha256"]:
            problemas.append(f"{arquivo}: identidade de desenvolvimento ausente ou diferente da do manifesto")
            continue
        cache = carregar_cache(pasta, extracao)
        diferentes = g7.diferencas_do_estudo(json.loads(arquivo.read_text(encoding="utf-8")), cache["identidade"])
        if diferentes:
            problemas.append(f"cache dos estudos {chave} difere do de desenvolvimento em {diferentes}")
        if cache["identidade"].get("tabela_sha256_conteudo") != conteudo_oficial:
            problemas.append(f"cache dos estudos {chave}: a tabela extraida nao e a oficial (membership + pb_examples)"
                             f": {g7.diferencas_de_tabela(oficial, cache['tabela']) or 'conteudo diferente'}")
        carregados[chave] = cache
        registro[chave] = {"pasta": str(pasta), "identidade_sha256": sha256_do_arquivo(Path(pasta) / "identidade.json")}
    if problemas:
        return {}, registro, problemas
    pontos = {}
    for sistema, nome in ((estudos.BASE, "base"), (estudos.REGIONALIZADO, "regionalized")):
        series = []
        for componente in manifesto["sistemas"][nome]["componentes"]:
            caminho = raiz / componente["arquivo"]
            if sha256_do_arquivo(caminho) != componente["sha256"]:
                problemas.append(f"{caminho}: sha256 diferente do manifesto")
                continue
            cache = carregados[componente["cache"]["chave"]]
            _, probabilidade = pontuar_salva(carregar_cabeca_salva(caminho), cache["matriz"])
            series.append(pd.Series(probabilidade, index=cache["tabela"]["variant_id"].astype(str).to_numpy()))
        if len(series) == 3:
            indice = series[0].index
            pontos[sistema] = pd.Series(g6.media_dos_componentes([s.reindex(indice).to_numpy() for s in series]),
                                        index=indice)
    return pontos, registro, problemas


def _n(valor: Any, formato: str = ".4f") -> str:
    return "-" if valor is None else format(valor, formato)


def _condicoes(regra: dict[str, Any]) -> str:
    if "condicoes" not in regra:
        return ""
    return " (" + "; ".join(f"{c['estatistica']} {_n(c['valor'], '+.4f')} {c['comparacao']} {c['limite']}: "
                            f"{c['atende']}" for c in regra["condicoes"]) + ")"


def imprimir(relatorio: dict[str, Any]) -> None:
    prefixo = "[ENSAIO] " if relatorio["modo"] == "ENSAIO" else ""
    if prefixo:
        print(f"\n{AVISO_DO_ENSAIO}")
    oficial, boot = relatorio["tabela_oficial"], relatorio["bootstrap"]
    print(f"{prefixo}tabela oficial: {oficial['variantes']:,} variantes ({oficial['no_chr8']} no chr8), conteudo "
          f"{oficial['conteudo_sha256'][:12]} | bootstrap: {boot['replicas']} replicas, seed {boot['seed']}, unidade "
          f"principal {boot['unidade_principal_da_interacao']} ({boot['origem']})")
    for estudo, bloco in relatorio["estudos"].items():
        sistemas = bloco["sistemas"]
        print(f"\n{prefixo}== {estudo} == pareamento {sistemas['pareamento']}")
        for coorte in (estudos.COORTE_COMPLETO, estudos.CASOS_PAREADOS, estudos.CONTROLES):
            c = sistemas["coortes"][coorte]["coorte"]
            linha = f"  {coorte:<14} n {c['intersecao']['n_total']:>5} (P {c['intersecao']['n_P_total']}, " \
                    f"B {c['intersecao']['n_B_total']}, cobertura {_n(c['intersecao']['coverage'], '.3f')})"
            print(linha)
            for metrica in ("auroc", "auprc", "brier"):
                if metrica not in c["delta"]:
                    continue
                d = c["delta"][metrica]
                print(f"    {metrica:<6} M0 {_n(c['base'][metrica]['estimativa'])}  MR "
                      f"{_n(c['regionalized'][metrica]['estimativa'])}  delta {_n(d['estimativa'], '+.4f')} "
                      f"[{_n(d['p2_5'], '+.4f')}; {_n(d['p97_5'], '+.4f')}]"
                      + ("  (Brier: negativo = melhor)" if metrica == "brier" else ""))
        principal = sistemas["interacao"]["reamostragem"]["unidade_principal"]
        for metrica in ("auroc", "auprc"):
            i = sistemas["interacao"][metrica]
            faixas = " | ".join(
                f"{unidade}{' (principal)' if unidade == principal else ''} [{_n(v['interacao']['p2_5'], '+.4f')}; "
                f"{_n(v['interacao']['p97_5'], '+.4f')}]" for unidade, v in i["por_unidade"].items())
            print(f"  interacao {metrica}: {_n(i['interacao']['estimativa'], '+.4f')} | {faixas}"
                  + ("" if principal else " (unidade principal nao declarada)"))
        partes = []
        for nome, s in sistemas["sensibilidades"].items():
            if "nao_calculada" in s:
                partes.append(f"{nome}: nao calculada ({s['nao_calculada']})")
            elif "nao_necessaria" in s:
                partes.append(f"{nome}: nao necessaria")
            else:
                partes.append(f"{nome}: {s['pares_mantidos']} pares mantidos, {s['pares_retirados']} retirados")
        print("  sensibilidades: " + " | ".join(partes))
        for nome, b in bloco["baselines"].items():
            if "nao_aplicavel" in b:
                print(f"  baseline {nome}: nao aplicavel -- {b['nao_aplicavel']}")
                continue
            partes = []
            for coorte in (estudos.COORTE_COMPLETO, estudos.CASOS_PAREADOS, estudos.CONTROLES):
                c = b["coortes"][coorte]["coorte"]
                partes.append(f"{coorte} {_n(c['auroc']['estimativa'])}" + (" (constante)" if c["constante"] else ""))
            print(f"  baseline {nome} (AUROC): " + " | ".join(partes))
    margens = relatorio["margens"]
    if not margens.get("avaliado"):
        print(f"\n{prefixo}margens: nao avaliadas -- {margens.get('motivo')}")
        return
    for estudo, bloco in margens["por_estudo"].items():
        print(f"\n{prefixo}margens em {estudo}: atende todas = {bloco['atende_todas']}"
              + (f" ({bloco['nota']})" if bloco.get("nota") else ""))
        for nome, regra in bloco["regras"].items():
            print(f"  {nome}: atende = {regra.get('atende')}{_condicoes(regra)}"
                  + (f" -- {regra['motivo']}" if regra.get("motivo") else ""))
            for rotulo, parte in {**regra.get("por_painel", {}), **regra.get("sem_cada_painel", {})}.items():
                print(f"      {rotulo}: atende = {parte['atende']}{_condicoes(parte)}")
    sucesso = margens["sucesso"]
    print(f"\n{prefixo}SUCESSO (estudos exigidos {sucesso['estudos_exigidos']}; descritivos "
          f"{sucesso['estudos_descritivos']}): {sucesso['atende']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ensaio-sintetico", action="store_true")
    parser.add_argument("--manifesto", type=Path, help="pasta do G6 congelado (modo real)")
    parser.add_argument("--raiz", type=Path, help="pasta dos comparadores (modo real)")
    parser.add_argument("--cache-estudos", action="append", default=[], help="M0=pasta ou semente=pasta (modo real)")
    parser.add_argument("--release-root", required=True, type=Path)
    parser.add_argument("--mosaic-root", required=True, type=Path)
    parser.add_argument("--entrada", action="append", default=[], help="regra_ampla=arquivo, exposicao=arquivo")
    parser.add_argument("--campanha", type=Path, default=RAIZ / "configs" / "campanha_r03_desenvolvimento.json")
    parser.add_argument("--replicas", type=int, help="so no ensaio (padrao 1000); no modo real, as do manifesto")
    parser.add_argument("--seed", type=int, help="so no ensaio (padrao 20260901); no modo real, a do manifesto")
    parser.add_argument("--seed-do-ensaio", type=int, default=20260925)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args(argv)

    inicio = time.perf_counter()
    destino = args.out_dir.expanduser()
    if destino.exists():
        print(f"FALHOU: {destino} ja existe; o G7 grava sempre numa pasta nova")
        return 2
    real = not args.ensaio_sintetico
    if real and (args.manifesto is None or args.raiz is None or not args.cache_estudos):
        print("FALHOU: o modo real exige --manifesto, --raiz e --cache-estudos (ou use --ensaio-sintetico)")
        return 2
    if args.ensaio_sintetico and (args.manifesto or args.cache_estudos):
        print("FALHOU: o ensaio nao le manifesto nem cache de estudo: e so a leitura dos artefatos, sem modelo")
        return 2
    problemas: list[str] = []
    manifesto, sha_do_manifesto = None, None
    campanha = carregar_campanha(args.campanha)
    if real:
        try:
            manifesto, sha_do_manifesto = g6.ler_manifesto_congelado(args.manifesto)
        except g6.ManifestoInvalido as exc:
            return _falhar([str(exc)])
        if sha256_do_arquivo(args.campanha.expanduser()) != manifesto["declaracao"]["sha256"]:
            problemas.append("a declaracao mudou depois do congelamento do G6")
        for arquivo, sha in manifesto["codigo"]["arquivos"].items():
            if not (RAIZ / arquivo).exists() or sha256_do_arquivo(RAIZ / arquivo) != sha:
                problemas.append(f"codigo diferente do congelado: {arquivo}")
        problemas += g6.problemas_do_codigo(estado_do_codigo(sorted(manifesto["codigo"]["arquivos"])))
        replicas, seed = int(manifesto["bootstrap"]["replicas"]), int(manifesto["bootstrap"]["seed"])
        for opcao, dado, congelado in (("--replicas", args.replicas, replicas), ("--seed", args.seed, seed)):
            if dado is not None and dado != congelado:
                problemas.append(f"{opcao} {dado} diverge do manifesto ({congelado}): o G7 real usa o congelado")
        unidade = manifesto["bootstrap"]["interacao"]["unidade_principal"]
        origem_do_bootstrap = "manifesto congelado"
        if problemas:
            return _falhar(problemas)
    else:
        replicas = args.replicas if args.replicas is not None else estudos.REPLICAS
        seed = args.seed if args.seed is not None else estudos.SEED
        bootstrap_declarado = campanha["g6"]["bootstrap_da_interacao"]
        proposta = bootstrap_declarado.get("unidade_principal")
        confirmada = not g6.problemas_do_bootstrap(bootstrap_declarado)
        unidade = proposta if confirmada else None
        origem_do_bootstrap = ("ensaio: linha de comando; unidade DECLARADA na declaracao" if confirmada else
                               f"ensaio: linha de comando; unidade ainda nao declarada (recomendada: {proposta})")

    logical_contract, especificacao, comparator_score, codigo_do_mosaic = cobertura.carregar_mosaic(
        args.mosaic_root.expanduser())
    problemas += cobertura.conferir_especificacao(especificacao)
    referencia = (manifesto["proveniencia"]["declarada"] if real else campanha["g6"]["proveniencia"])[
        "release_do_mosaic"]
    membros, anotacoes, oficial, problemas_do_release = ler_release(args.release_root.expanduser(), logical_contract,
                                                                    referencia)
    problemas += problemas_do_release
    regra_ampla, exposicao, hashes_das_entradas, problemas_das_entradas = ler_entradas(
        ler_pares(args.entrada, "--entrada"), manifesto["entradas_das_analises_secundarias"] if real else None)
    problemas += problemas_das_entradas
    if problemas:
        return _falhar(problemas)

    ids = set(membros["variant_id"].astype(str))
    scores_reais = baselines.scores_das_baselines(membros, anotacoes,
                                                  lambda linha: comparator_score(linha, especificacao))
    cobertura_das_baselines = {nome: {"pontuadas": int(np.isfinite(s.to_numpy()).sum()), "membros_unicos": len(s)}
                               for nome, s in scores_reais.items()}
    if real:
        pastas = {chave: Path(p).expanduser() for chave, p in ler_pares(args.cache_estudos, "--cache-estudos").items()}
        esperadas = {c["cache"]["chave"] for n in ("base", "regionalized") for c in manifesto["sistemas"][n]["componentes"]}
        if set(pastas) != esperadas:
            return _falhar([f"--cache-estudos {sorted(pastas)}, o manifesto pede {sorted(esperadas)}"])
        pontos, registro_dos_caches, problemas_dos_caches = pontuar_sistemas(manifesto, args.raiz.expanduser(), pastas,
                                                                             oficial)
        if problemas_dos_caches:
            return _falhar(problemas_dos_caches)
        scores = scores_reais
        limiares = g7.limiares_do_manifesto(manifesto)
        margens, bootstrap = manifesto["margens"], manifesto["bootstrap"]["interacao"]
    else:
        pontos = g7.pontos_sinteticos(membros, seed=args.seed_do_ensaio)
        scores = g7.baselines_sinteticas(membros, list(baselines.ESPECIFICACOES), seed=args.seed_do_ensaio + 1)
        registro_dos_caches = None
        limiares = {estudos.BASE: 0.5, estudos.REGIONALIZADO: 0.5}
        margens, bootstrap = campanha["g6"]["margens"], campanha["g6"]["bootstrap_da_interacao"]

    resultados, relatorio_dos_estudos = {}, {}
    for estudo in estudos.ESTUDOS:
        print(f"[g7] {estudo}: consumidor ({replicas} replicas, seed {seed}, unidade principal {unidade})",
              flush=True)
        resultados[estudo] = estudos.avaliar_estudo(
            membros, estudo, pontos, replicas=replicas, seed=seed, limiares=limiares,
            controles_com_scv_brasileira=regra_ampla, exposicao=exposicao_do_estudo(exposicao, estudo),
            tolerancia_de_exposicao=0, com_brier=True, unidade_principal=unidade,
            progresso=lambda texto: print(f"  {texto}", flush=True))
        print(f"[g7] {estudo}: baselines", flush=True)
        relatorio_dos_estudos[estudo] = {
            "sistemas": resultados[estudo],
            "baselines": {nome: estudos.avaliar_baseline(membros, estudo, scores[nome],
                                                         especificacao=baselines.ESPECIFICACOES[nome],
                                                         replicas=replicas, seed=seed)
                          for nome in baselines.ESPECIFICACOES}}
    relatorio = {
        "formato": "campanha_r03_g7_relatorio_v1", "modo": "REAL" if real else "ENSAIO",
        "aviso": None if real else AVISO_DO_ENSAIO,
        "manifesto": {"pasta": str(args.manifesto), "sha256": sha_do_manifesto} if real else None,
        "natureza": campanha["g6"]["natureza"],
        "release": {"referencia": referencia, "mosaic": codigo_do_mosaic},
        "entradas_das_analises_secundarias": hashes_das_entradas,
        "caches_dos_estudos": registro_dos_caches,
        "limiares": limiares if real else "ensaio: 0,5 fixo, sem significado",
        "leitura_do_chr8": g7.LEITURA_DO_CHR8,
        "tabela_oficial": {"variantes": int(len(oficial)), "no_chr8": int((oficial["chrom"] == "chr8").sum()),
                           "conteudo_sha256": hash_do_conteudo(oficial),
                           "origem": "membership + pb_examples do release, hash logico conferido"},
        "bootstrap": {"replicas": replicas, "seed": seed, "unidade_principal_da_interacao": unidade,
                      "origem": origem_do_bootstrap},
        "baselines": {"especificacoes": baselines.ESPECIFICACOES, "cobertura_do_score": cobertura_das_baselines,
                      "frequencia_observada": baselines.frequencia_observada(membros, anotacoes),
                      "leitura": "cobertura do score nao e frequencia observada: a regra de cada baseline diz o que "
                                 "recebeu imputacao"},
        "estudos": relatorio_dos_estudos,
        "margens": g7.avaliar_margens(resultados, margens, bootstrap),
        "segundos": round(time.perf_counter() - inicio, 1),
    }
    destino.mkdir(parents=True)
    nome = "g7_relatorio.json" if real else "ensaio_relatorio.json"
    (destino / nome).write_text(json.dumps(relatorio, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    if real:
        tabela = pd.DataFrame({"variant_id": sorted(ids)})
        for sistema, serie in pontos.items():
            tabela[f"prob_{sistema}"] = serie.reindex(tabela["variant_id"]).to_numpy()
        for nome_da_baseline, serie in scores.items():
            tabela[f"baseline_{nome_da_baseline}"] = serie.reindex(tabela["variant_id"]).to_numpy()
        tabela.to_parquet(destino / "g7_pontos.parquet", index=False)
    imprimir(relatorio)
    print(f"\n{'PASSOU (ENSAIO)' if not real else 'G7 AVALIADO'}: relatorio em {destino / nome}")
    return 0


def _falhar(problemas: list[str]) -> int:
    print(f"FALHOU: {len(problemas)} problema(s)")
    for problema in problemas:
        print(f"  - {problema}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
