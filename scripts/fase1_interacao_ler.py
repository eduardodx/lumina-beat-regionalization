#!/usr/bin/env python3
"""Fase 1, cabeca com interacao: leituras de E+F, E+F+BR2, H(E+F) e H(E+F+BR2) (desenvolvimento exploratorio,
posterior ao teste).

Especificacao, congelada antes de rodar: docs/fase1_cabeca_interacao_especificacao.md. Mede com o MESMO codigo do
passo 4, do diagnostico e do BR v2, nos pares (delta = novo - base):
- E+F -> H(E+F): capacidade, sem ABraOM (relatado com perdas e ganhos);
- H(E+F) -> H(E+F+BR2): a informacao regional com a cabeca com interacao;
- E+F -> H(E+F+BR2): seguranca do sistema novo contra a referencia original;
- E+F+BR2 -> H(E+F+BR2): a cabeca com o BR2 presente;
- E+F -> E+F+BR2: reproducao. As perdas, os ganhos e os pontos de AUROC brasileiros tem de ser os do br2.json.

Saem: nucleo, proxies, beneficio, P-BR no limiar MCC e na especificidade equivalente (perdas e ganhos separados), as
P-BR perdidas pelo BR2 uma a uma (com E sozinho, do passo 3, para contexto), as criticas, os coeficientes de S e das
interacoes (leitura) e o veredito da leitura declarada, calculado pelas regras da especificacao. Nada aqui e
confirmatorio.

USO (notebook, no .venv do Mosaic)
    PYTHONPATH="$PWD" uv run --project ~/mosaic-v1-2026-09-30 --frozen python scripts/fase1_interacao_ler.py \\
        --entrega ~/mosaic-v1-2026-09-30 --bracos <passo 3> --leituras <passo 4> --br2 <treino do BR v2> \\
        --leituras-br2 <leituras do BR v2> --h <treino da cabeca H> --avaliacao-h <avaliador nos bracos H> \\
        --out-dir <nova>

SAIDAS (em --out-dir, que nao pode existir): interacao.json, interacao.md e perdas_e_ganhos.csv.
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

from eval.campanha.cache import sha256_do_arquivo  # noqa: E402
from scripts import fase1_br2_ler as ler_br2  # noqa: E402
from scripts import fase1_br2_treinar as treinar_br2  # noqa: E402
from scripts import fase1_bracos as bracos  # noqa: E402
from scripts import fase1_diagnostico as diag  # noqa: E402
from scripts import fase1_interacao_treinar as treinar_h  # noqa: E402
from scripts import fase1_leituras as leituras  # noqa: E402
from scripts.inventario_mosaic_v1 import K, ler_criticas  # noqa: E402

FalhaDaFase1 = bracos.FalhaDaFase1
H_EF, H_EFBR2, BR2 = "H(E+F)", "H(E+F+BR2)", treinar_br2.BRACO
IDS = {"E+F": bracos.BRACOS["E+F"]["id"], BR2: treinar_br2.ID,
       **{braco: cfg["id"] for braco, cfg in treinar_h.BRACOS_H.items()}}
PARES_H: tuple[tuple[str, str], ...] = (("E+F", H_EF), (H_EF, H_EFBR2), ("E+F", H_EFBR2), (BR2, H_EFBR2),
                                        ("E+F", BR2))
CAPACIDADE, REGIONAL, SEGURANCA, CABECA, REPRODUCAO = (leituras.nome_do_par(b, n) for b, n in PARES_H)
#: A leitura declarada (docs/fase1_cabeca_interacao_especificacao.md). Cada numero vale so para a sua leitura.
DECLARADA = {"perdas_mcc_max": 6, "perdas_especificidade_max": 5, "protegidas_min": 6, "contra_perdas_mcc_min": 10,
             "contra_protegidas_max": 3, "contrastes": [REGIONAL, SEGURANCA]}


# --------------------------------------------------------------------------------------------- leitura declarada

def veredito(perdas: dict[str, dict[str, int]], protegidas: int, delta_auroc: dict[str, dict[str, float]],
             falso_positivo: dict[str, float]) -> dict[str, Any]:
    """As regras da especificacao. `perdas[contraste]` = {"mcc": n, "especificidade": n} para REGIONAL e SEGURANCA;
    `protegidas` = quantas das P-BR perdidas pelo BR2 H(E+F+BR2) chama; `delta_auroc` = {"regional": {...},
    "referencia": {...}} com "clinico" e "beneficio"; `falso_positivo` = taxa no recorte de beneficio por braco H."""
    d = DECLARADA
    apoio = {
        "1_perdas_nos_dois_contrastes": all(perdas[c]["mcc"] <= d["perdas_mcc_max"]
                                            and perdas[c]["especificidade"] <= d["perdas_especificidade_max"]
                                            for c in d["contrastes"]),
        "2_protegidas": protegidas >= d["protegidas_min"],
        "3_recortes_brasileiros": all(delta_auroc["regional"][k] >= delta_auroc["referencia"][k] - 1e-12
                                      for k in ("clinico", "beneficio")),
        "4_falso_positivo_das_benignas_no_abraom": falso_positivo[H_EFBR2] <= falso_positivo[H_EF] + 1e-12,
    }
    contra = {
        "perdas_mcc_em_algum_contraste": any(perdas[c]["mcc"] >= d["contra_perdas_mcc_min"] for c in d["contrastes"]),
        "protegidas_no_maximo_3": protegidas <= d["contra_protegidas_max"],
    }
    resultado = "contra" if any(contra.values()) else "apoio" if all(apoio.values()) else "inconclusivo"
    return {"veredito": resultado, "criterios_de_apoio": apoio, "criterios_contra": contra, "regras": d,
            "entradas": {"perdas": perdas, "protegidas": protegidas, "delta_auroc": delta_auroc,
                         "falso_positivo": falso_positivo},
            "natureza": "orientacao de desenvolvimento, nao criterio de avanco demonstrado"}


def _ponto(bloco: dict[str, Any]) -> float:
    return float(bloco["coorte"]["delta"]["auroc"]["estimativa"])


def pontos_brasileiros(proxies: dict[str, Any], beneficio: dict[str, Any], par: str) -> dict[str, float]:
    return {"clinico": _ponto(proxies[par]["br_clinical_evidence"]["coortes"]["full_cohort"]),
            "beneficio": _ponto(beneficio["pares"][par]["continuas"])}


def conferir_reproducao(lido: dict[str, Any], br2: dict[str, Any]) -> dict[str, Any]:
    """E+F -> E+F+BR2 tem de reproduzir as perdas, os ganhos e os pontos de AUROC brasileiros do br2.json."""
    novo = ler_br2.NOVO
    for chave in ("ids_perdidas", "ids_ganhas"):
        if set(lido["p_br"]["pares"][REPRODUCAO][chave]) != set(br2["p_br"]["pares"][novo][chave]):
            raise FalhaDaFase1(f"{REPRODUCAO}: {chave} nao reproduz as leituras do BR v2")
    aqui = pontos_brasileiros(lido["proxies"], lido["beneficio"], REPRODUCAO)
    la = pontos_brasileiros(br2["proxies"], br2["beneficio"], novo)
    if any(not np.isclose(aqui[k], la[k], rtol=0, atol=1e-9) for k in aqui):
        raise FalhaDaFase1(f"{REPRODUCAO}: pontos de AUROC brasileiros diferentes dos do BR v2 ({aqui} x {la})")
    return {"p_br": "igual ao br2.json", "pontos_brasileiros": aqui}


def perdas_do_br2(pbr: pd.DataFrame, ids: list[str], chamadas: dict[str, pd.DataFrame],
                  benignas: dict[str, dict[int, np.ndarray]]) -> list[dict[str, Any]]:
    """As P-BR perdidas por E+F -> E+F+BR2, uma a uma: chamada e fpr exigido em cada braco."""
    saida = []
    for vid in ids:
        linha = pbr.loc[vid]
        gene = next((v for v in (linha.get("mane_gene"), linha.get("clinvar_gene_symbol")) if isinstance(v, str) and v),
                    "—")
        variante = next((v for v in (linha.get("aa_change"), linha.get("consequence")) if isinstance(v, str) and v),
                        "—")
        por_braco = {}
        for braco, c in chamadas.items():
            score, run = float(c.loc[vid, "score"]), int(c.loc[vid, "run"])
            por_braco[braco] = {"chamada": str(c.loc[vid, "chamada"]),
                                "fpr_exigido": diag.fpr_exigido(score, benignas[braco][run])}
        saida.append({"variant_id": vid, "gene": gene, "variante": variante, "tier": linha.get("label_tier"),
                      "abraom_ac": linha.get("abraom_ac"), "abraom_an": linha.get("abraom_an"), "bracos": por_braco})
    return saida


# ----------------------------------------------------------------------------------------------- relatorio

def _f(valor: Any, casas: int = 3) -> str:
    return "—" if valor is None or (isinstance(valor, float) and not np.isfinite(valor)) else f"{valor:.{casas}f}"


def secao_h(d: dict[str, Any]) -> str:
    v = d["veredito"]
    L = ["# Cabeça com interação: leitura declarada, perdas do BR2 e coeficientes", "",
         f"**Veredito pela leitura declarada: {v['veredito']}.** Orientação de desenvolvimento, não critério de avanço "
         "demonstrado; regras em docs/fase1_cabeca_interacao_especificacao.md.", "",
         "| critério | valor | cumpre |", "|---|---|---|"]
    e = v["entradas"]
    for c in DECLARADA["contrastes"]:
        L.append(f"| perdas de P-BR em {c} (MCC ≤ {DECLARADA['perdas_mcc_max']}; especificidade equivalente ≤ "
                 f"{DECLARADA['perdas_especificidade_max']}) | {e['perdas'][c]['mcc']}; {e['perdas'][c]['especificidade']} "
                 f"| {'sim' if e['perdas'][c]['mcc'] <= DECLARADA['perdas_mcc_max'] and e['perdas'][c]['especificidade'] <= DECLARADA['perdas_especificidade_max'] else 'não'} |")
    L += [f"| perdas do BR2 chamadas por {H_EFBR2} (≥ {DECLARADA['protegidas_min']}) | {e['protegidas']} de "
          f"{d['perdas_do_br2']['n']} | {'sim' if v['criterios_de_apoio']['2_protegidas'] else 'não'} |",
          f"| Δ AUROC de {REGIONAL}: clínico e benefício (não abaixo de {_f(e['delta_auroc']['referencia']['clinico'], 4)} "
          f"e {_f(e['delta_auroc']['referencia']['beneficio'], 4)}) | {_f(e['delta_auroc']['regional']['clinico'], 4)}; "
          f"{_f(e['delta_auroc']['regional']['beneficio'], 4)} | "
          f"{'sim' if v['criterios_de_apoio']['3_recortes_brasileiros'] else 'não'} |",
          f"| falso-positivo nas benignas presentes no ABraOM: {H_EFBR2} ≤ {H_EF} | "
          f"{_f(e['falso_positivo'][H_EFBR2], 4)}; {_f(e['falso_positivo'][H_EF], 4)} | "
          f"{'sim' if v['criterios_de_apoio']['4_falso_positivo_das_benignas_no_abraom'] else 'não'} |",
          "", f"Contra: {v['criterios_contra']}. A regra de segurança (margem 0,01) não muda; ver a tabela de P-BR.", "",
          f"## As {d['perdas_do_br2']['n']} P-BR perdidas por E+F -> E+F+BR2 (fpr exigido e chamada)", "",
          "| gene | variante | tier | AC/AN | " + " | ".join(d["perdas_do_br2"]["bracos"]) + " |",
          "|---|---|---|---|" + "---|" * len(d["perdas_do_br2"]["bracos"])]
    for r in d["perdas_do_br2"]["variantes"]:
        acan = "—" if r["abraom_ac"] is None or pd.isna(r["abraom_ac"]) else f"{int(r['abraom_ac'])}/{int(r['abraom_an'])}"
        L.append(f"| {r['gene']} | {r['variante']} | {r['tier']} | {acan} | "
                 + " | ".join(f"{_f(r['bracos'][b]['fpr_exigido'])} ({'+' if r['bracos'][b]['chamada'] == 'positive' else '−'})"
                              for b in d["perdas_do_br2"]["bracos"]) + " |")
    L += ["", "## Coeficientes médios de S e das interações (features padronizadas; só leitura)", ""]
    for braco, coef in d["coeficientes"].items():
        L.append(f"- **{braco}:** " + "; ".join(f"{c} {valor:+.3f}" for c, valor in coef["media"].items()))
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------------------------------------- main

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--entrega", required=True, type=Path)
    parser.add_argument("--bracos", required=True, type=Path, help="pasta do passo 3 (E+F e E)")
    parser.add_argument("--leituras", required=True, type=Path, help="pasta do passo 4 (limiares de E e E+F)")
    parser.add_argument("--br2", required=True, type=Path, help="pasta do treino do BR v2")
    parser.add_argument("--leituras-br2", required=True, type=Path, help="pasta das leituras do BR v2 (br2.json)")
    parser.add_argument("--h", required=True, type=Path, help="pasta do treino da cabeca H")
    parser.add_argument("--avaliacao-h", type=Path, help="--output-root do evaluate_candidate.py nos bracos H")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--replicas", type=int, default=leituras.REPLICAS)
    args = parser.parse_args(argv)
    entrega, pasta3, pasta4 = args.entrega.expanduser(), args.bracos.expanduser(), args.leituras.expanduser()
    pasta_br2, pasta_l2, pasta_h = args.br2.expanduser(), args.leituras_br2.expanduser(), args.h.expanduser()
    destino = args.out_dir.expanduser()
    if destino.exists():
        return leituras._falhar([f"{destino} ja existe; as leituras da cabeca H gravam sempre numa pasta nova"])
    try:
        ferramentas = leituras.ferramentas_do_mosaic()
    except ImportError as exc:
        return leituras._falhar([f"sem o pacote mosaic ({exc}); rodar no .venv do Mosaic (uv run --project)"])
    inicio = time.perf_counter()

    def avisar(texto: str) -> None:
        print(f"[{time.perf_counter() - inicio:7.1f} s] {texto}", flush=True)

    raiz = entrega / bracos.RELEASE
    try:
        identidade = ferramentas["verificar_identidade"](raiz, config_dir=entrega / "config")
        bracos.conferir_entrega(identidade, sha256_do_arquivo(entrega / bracos.PROTOCOLO_DE_ESTUDOS))
        release = leituras.carregar_release(raiz)
        release = release.merge(pd.read_parquet(raiz / "variant-annotations.parquet", columns=["variant_id", "abraom_ac"]),
                                on="variant_id", how="left", validate="one_to_one")
        for pasta in (pasta3, pasta_br2, pasta_h):
            leituras.conferir_fontes(json.loads((pasta / "fontes.json").read_text(encoding="utf-8")), raiz)
            leituras.conferir_selecao(json.loads((pasta / "selecao.json").read_text(encoding="utf-8")))
        lidas4 = json.loads((pasta4 / "leituras.json").read_text(encoding="utf-8"))
        br2 = json.loads((pasta_l2 / "br2.json").read_text(encoding="utf-8"))
        fontes_h = json.loads((pasta_h / "fontes.json").read_text(encoding="utf-8"))
        if Path(lidas4["proveniencia"]["bracos"]).expanduser().resolve() != pasta3.resolve():
            raise FalhaDaFase1(f"o passo 4 leu outro passo 3: {lidas4['proveniencia']['bracos']}")
        if Path(fontes_h["passo3"]).expanduser().resolve() != pasta3.resolve():
            raise FalhaDaFase1(f"a cabeca H usou outro passo 3: {fontes_h['passo3']}")
        if fontes_h.get("especificacao_sha256") != sha256_do_arquivo(treinar_h.ESPECIFICACAO):
            raise FalhaDaFase1("a especificacao mudou depois do treino da cabeca H")
        tres = leituras.carregar_predicoes(pasta3)
        preds = {"E+F": tres["E+F"]}
        for braco, pasta, ident in ((BR2, pasta_br2, treinar_br2.ID),
                                    *((b, pasta_h, cfg["id"]) for b, cfg in treinar_h.BRACOS_H.items())):
            p = pd.read_parquet(pasta / ident / "predictions.parquet")
            preds[braco] = p.assign(variant_id=p["variant_id"].astype(str), run=p["run"].astype(int))
        conferencia_das_linhas = leituras.conferir_linhas(preds, release)
        rotulo = release.set_index("variant_id")["binary_label"].astype(int)
        limiares = leituras.limiares(preds, rotulo, ferramentas["calibrate_threshold"])
        for braco, gravado in (("E+F", lidas4["limiares"]["E+F"]), (BR2, br2["limiares"][BR2])):
            if any(not np.isclose(float(gravado[str(r)]), limiares[braco][r], rtol=0, atol=1e-12) for r in range(K)):
                raise FalhaDaFase1(f"{braco}: limiares diferentes dos gravados no passo 4 ou no BR v2")
        oficial = (leituras.avaliador_oficial(args.avaliacao_h.expanduser(), {b: limiares[b] for b in treinar_h.BRACOS_H},
                                              ids=IDS) if args.avaliacao_h else None)
        membros = pd.read_parquet(raiz / leituras.MEMBERSHIP)
        lista_de_criticas = ler_criticas(entrega / leituras.CRITICAS)
        limiar_e = {int(r): float(t) for r, t in lidas4["limiares"]["E"].items()}
    except (FalhaDaFase1, ValueError, KeyError, OSError, RuntimeError, AssertionError) as exc:
        return leituras._falhar([f"{type(exc).__name__}: {exc}"])
    avisar(f"entradas conferidas: {conferencia_das_linhas['linhas_por_braco']:,} linhas por braco; limiares de E+F e "
           f"E+F+BR2 iguais aos gravados; {'limiares dos bracos H conferidos com o avaliador' if oficial else 'sem avaliador'}")

    chamadas = {braco: leituras.chamadas_de_teste(p, limiares[braco]) for braco, p in preds.items()}
    try:
        lido: dict[str, Any] = {
            "formato": "fase1_interacao_v1",
            "declaracao": {"pares": [leituras.nome_do_par(b, n) for b, n in PARES_H], "delta": "novo - base",
                           "replicas": args.replicas, "seed": leituras.SEED, "leitura_declarada": DECLARADA,
                           "natureza": "desenvolvimento exploratorio posterior ao teste; especificacao em "
                                       "docs/fase1_cabeca_interacao_especificacao.md"},
            "limiares": {b: {str(r): t for r, t in v.items()} for b, v in limiares.items()},
            "avaliador_oficial": oficial,
        }
        lido["nucleo"] = leituras.nucleo(ferramentas, raiz, release, preds, replicas=args.replicas, seed=leituras.SEED,
                                         avisar=avisar, pares=PARES_H, descritivos=(), ids=IDS)
        lido["beneficio"] = leituras.beneficio(ferramentas, raiz, release, preds, chamadas, replicas=args.replicas,
                                               seed=leituras.SEED, avisar=avisar, pares=PARES_H, ids_dos_sistemas=IDS)
        lido["p_br"] = leituras.p_br(release, chamadas, lista_de_criticas, pares=PARES_H)
        lido["criticas"] = leituras.criticas_por_braco(lista_de_criticas, release, chamadas, pares=PARES_H)
        pbr = diag.carregar_p_br(raiz)
        benignas = diag.benignas_da_validacao(preds, rotulo)
        diagnostico, tabelas = diag.diagnosticar(pbr, chamadas, benignas, PARES_H, lido["p_br"]["pares"])
        lido["diagnostico"] = {"bracos": list(chamadas), "pares": diagnostico,
                               "criticas": diag.criticas_com_fpr(lista_de_criticas, raiz, chamadas, benignas)}
        avisar("proxies (o mais demorado)")
        lido["proxies"] = leituras.proxies(membros, preds, replicas=args.replicas, seed=leituras.SEED, avisar=avisar,
                                           pares=PARES_H)
        lido["reproducao"] = conferir_reproducao(lido, br2)
        com_e = {"E": leituras.chamadas_de_teste(tres["E"], limiar_e), **chamadas}
        ids_br2 = sorted(br2["p_br"]["pares"][ler_br2.NOVO]["ids_perdidas"])
        variantes = perdas_do_br2(pbr, ids_br2, com_e, {**diag.benignas_da_validacao({"E": tres["E"]}, rotulo),
                                                       **benignas})
        protegidas = sum(r["bracos"][H_EFBR2]["chamada"] == "positive" for r in variantes)
        lido["perdas_do_br2"] = {"n": len(ids_br2), "bracos": list(com_e), "variantes": variantes,
                                 "chamadas_por": {b: sum(r["bracos"][b]["chamada"] == "positive" for r in variantes)
                                                  for b in com_e}}
        lido["coeficientes"] = {braco: ler_br2.coeficientes(pasta_h, cfg["id"], "s_")
                                for braco, cfg in treinar_h.BRACOS_H.items()}
        perdas = {c: {"mcc": int(lido["p_br"]["pares"][c]["perdidas"]),
                      "especificidade": int(diagnostico[c]["especificidade_equivalente"]["perdidas"])}
                  for c in DECLARADA["contrastes"]}
        lido["veredito"] = veredito(
            perdas, int(protegidas),
            {"regional": pontos_brasileiros(lido["proxies"], lido["beneficio"], REGIONAL),
             "referencia": lido["reproducao"]["pontos_brasileiros"]},
            {b: float(lido["beneficio"]["por_braco"][b]["false_positive_rate"]) for b in (H_EF, H_EFBR2)})
    except (FalhaDaFase1, leituras.estudos.EstudoInvalido, ValueError, KeyError, TypeError) as exc:
        return leituras._falhar([f"{type(exc).__name__}: {exc}"])
    lido["proveniencia"] = {
        "entrega": str(entrega),
        "bracos": f"{pasta3} (E+F, E), {pasta_br2} ({BR2}) e {pasta_h} ({H_EF}, {H_EFBR2})",
        "leituras": str(pasta4), "leituras_br2": str(pasta_l2), "especificacao_sha256": fontes_h["especificacao_sha256"],
        "script_sha256": sha256_do_arquivo(Path(__file__)), "revisao": bracos.revisao_do_repositorio(),
        "versoes": bracos.versoes(), "segundos": round(time.perf_counter() - inicio, 1)}

    destino.mkdir(parents=True)
    (destino / "interacao.json").write_text(json.dumps(lido, indent=2, ensure_ascii=False, default=leituras._json),
                                            encoding="utf-8")
    relido = json.loads((destino / "interacao.json").read_text(encoding="utf-8"))
    texto = (leituras.resumo(relido, titulo="# Fase 1, cabeça com interação: leituras (desenvolvimento, posterior ao teste)")
             + "\n" + diag.relatorio(relido["diagnostico"], titulo="# Cabeça com interação: perdas e ganhos de P-BR")
             + "\n" + secao_h(relido))
    (destino / "interacao.md").write_text(texto, encoding="utf-8")
    pd.concat(tabelas, ignore_index=True).drop(columns=["core_purged_runs"], errors="ignore").to_csv(
        destino / "perdas_e_ganhos.csv", index=False)
    leituras._imprimir(texto)
    print(f"PASSOU: leituras da cabeca H em {destino}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
