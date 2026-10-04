#!/usr/bin/env python3
"""Fase 1, diagnostico posterior ao teste: as perdas e os ganhos de P-BR entre bracos e as criticas.

Desenvolvimento exploratorio definido DEPOIS de ler o teste (docs/fase1_r03_congelado.md, proximos passos). Nao muda
o resultado do passo 4 nem a regra de seguranca: so le as predicoes do passo 3, os limiares e as perdas do passo 4 e o
release, e confere que reproduz as perdas e os ganhos que o passo 4 gravou.

Para cada par do passo 4 (base -> novo):
- cada P-BR perdida (positiva na base, negativa no novo) e cada ganha, com gene, painel, tier, revisao do ClinVar,
  frequencias, filtros e estados no gnomAD e no ABraOM, execucao e, em cada braco, score, limiar e margem;
- o falso-positivo de validation que a chamada exigiria (`fpr_exigido`): a fracao das benignas da validation da
  execucao com score >= o da variante; ao lado, o falso-positivo no limiar de cada braco (`fpr_no_limiar`);
- o mecanismo, descritivo. Uma perda e de PONTO DE OPERACAO quando o novo ainda chamaria a variante no
  falso-positivo de validation da base; caso contrario, e de ORDENACAO. Um ganho e de ponto de operacao quando a base
  tambem a chamaria no falso-positivo do novo;
- a recontagem em ESPECIFICIDADE EQUIVALENTE (analise posterior): em cada execucao, o limiar do novo e refeito na
  validation para a especificidade que a base tem no proprio limiar, e as P-BR sao recontadas.
E as 13 criticas: as do release com o fpr_exigido em cada braco, as de fora marcadas.

USO (notebook, no .venv do Mosaic)
    PYTHONPATH="$PWD" uv run --project ~/mosaic-v1-2026-09-30 --frozen python scripts/fase1_diagnostico.py \\
        --entrega ~/mosaic-v1-2026-09-30 --bracos ~/artifacts/mosaic_v1/bracos_<...> \\
        --leituras ~/artifacts/mosaic_v1/leituras_<...> --out-dir ~/artifacts/mosaic_v1/diagnostico_<...>

SAIDAS (em --out-dir, que nao pode existir): diagnostico.json, diagnostico.md e perdas_e_ganhos.csv (uma linha por
variante e par).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from eval.campanha.cache import sha256_do_arquivo  # noqa: E402
from scripts import fase1_bracos as bracos  # noqa: E402
from scripts import fase1_leituras as leituras  # noqa: E402
from scripts.inventario_mosaic_v1 import K, ler_criticas  # noqa: E402
from scripts.inventario_mosaic_v1 import criticas as criticas_do_release  # noqa: E402

FalhaDaFase1 = bracos.FalhaDaFase1
ANOTACOES = ("mane_gene", "clinvar_gene_symbol", "consequence", "aa_change", "gnomad_v4_af", "gnomad_v4_popmax_af",
             "gnomad_v4_af_afr", "gnomad_v4_af_amr", "gnomad_v4_af_nfe", "gnomad_v4_faf95", "gnomad_v4_fafmax_faf95",
             "gnomad_v4_filter", "gnomad_status", "abraom_af", "abraom_ac", "abraom_an", "abraom_filter",
             "abraom_status", "phylop_241way")
CLINICAS = ("aggregate_classification", "aggregate_review_status", "br_lab_any")
#: Faixas de frequencia alelica (limite inferior incluido); 0 e ausencia ficam numa faixa propria.
FAIXAS = ((0.0, 1e-3, "<0,1%"), (1e-3, 1e-2, "0,1-1%"), (1e-2, 5e-2, "1-5%"), (5e-2, 1.01, ">=5%"))


# ------------------------------------------------------------------------------------------------- entradas

def colunas_presentes(caminho: Path, desejadas: tuple[str, ...]) -> list[str]:
    import pyarrow.parquet as pq

    nomes = set(pq.read_schema(caminho).names)
    return [c for c in desejadas if c in nomes]


def carregar_p_br(raiz: Path) -> pd.DataFrame:
    """As P-BR do passo 4 (P/LP presentes no ABraOM, gold e consensus) com as colunas descritivas que o release tiver."""
    release = leituras.carregar_release(raiz)
    pbr = release[(release["binary_label"] == 1) & release["present_abraom"]]
    extras = colunas_presentes(raiz / "variant-annotations.parquet", ANOTACOES)
    extras = [c for c in extras if c not in pbr.columns]
    if extras:
        pbr = pbr.merge(pd.read_parquet(raiz / "variant-annotations.parquet", columns=["variant_id", *extras]),
                        on="variant_id", how="left", validate="one_to_one")
    clinicas = [c for c in colunas_presentes(raiz / "clinical-variants.parquet", CLINICAS) if c not in pbr.columns]
    if clinicas:
        pbr = pbr.merge(pd.read_parquet(raiz / "clinical-variants.parquet", columns=["variant_id", *clinicas]),
                        on="variant_id", how="left", validate="one_to_one")
    pbr = pbr.assign(run=pbr["core_fold"].astype(int))   # a execucao que testa a variante
    return pbr.set_index("variant_id", drop=False)


def benignas_da_validacao(preds: dict[str, pd.DataFrame], rotulo: pd.Series) -> dict[str, dict[int, np.ndarray]]:
    """Scores ordenados das benignas da validation de cada braco e execucao (as linhas onde o limiar foi escolhido)."""
    saida: dict[str, dict[int, np.ndarray]] = {}
    for braco, p in preds.items():
        v = p[p["role"] == "validation"]
        y = rotulo.reindex(v["variant_id"]).to_numpy()
        saida[braco] = {run: np.sort(v["score"].to_numpy(dtype=float)[(v["run"].to_numpy() == run) & (y == 0)])
                        for run in range(K)}
    return saida


# --------------------------------------------------------------------------------------------- quantidades

def fpr_exigido(score: float, benignas: np.ndarray) -> float:
    """A fracao das benignas da validation com score >= `score`: o falso-positivo de validation de um limiar no score."""
    return float((len(benignas) - np.searchsorted(benignas, score, side="left")) / len(benignas))


def limiar_com_especificidade(benignas: np.ndarray, alvo: float) -> float:
    """O menor limiar (regra `score >= limiar`) cuja especificidade na validation e >= `alvo`: logo acima da benigna
    que completa as que tem de ficar abaixo. E o mesmo criterio do mecanismo: uma variante com `fpr_exigido` no
    falso-positivo alvo fica positiva com este limiar."""
    precisa = math.ceil(alvo * len(benignas) - 1e-9)   # benignas que tem de ficar abaixo do limiar
    if precisa <= 0:
        return float(np.nextafter(benignas[0], -np.inf))
    distintos = np.unique(benignas)
    ate = np.searchsorted(benignas, distintos, side="right")   # benignas <= cada valor
    return float(np.nextafter(distintos[np.nonzero(ate >= precisa)[0][0]], np.inf))


def _faixa(af: Any) -> str:
    if af is None or pd.isna(af) or af <= 0:
        return "0 ou ausente"
    return next(nome for baixo, alto, nome in FAIXAS if baixo <= af < alto)


def _razao(abraom: Any, gnomad: Any) -> str:
    if gnomad is None or pd.isna(gnomad) or gnomad <= 0:
        return "gnomAD 0 ou ausente"
    r = float(abraom) / float(gnomad)
    return "<0,5x" if r < 0.5 else "0,5-2x" if r < 2 else "2-5x" if r < 5 else ">=5x"


def _contagem(serie: pd.Series) -> dict[str, int]:
    return {str(k): int(v) for k, v in serie.fillna("<nulo>").value_counts().sort_index().items()}


def _resumo_numerico(valores: pd.Series) -> dict[str, float | None]:
    v = pd.to_numeric(valores, errors="coerce").dropna()
    if v.empty:
        return {"n": 0, "mediana": None, "p10": None, "p90": None}
    return {"n": int(len(v)), "mediana": float(v.median()), "p10": float(v.quantile(0.1)),
            "p90": float(v.quantile(0.9))}


def linhas_do_par(pbr: pd.DataFrame, chamadas: dict[str, pd.DataFrame], benignas: dict[str, dict[int, np.ndarray]],
                  base: str, novo: str) -> pd.DataFrame:
    """Uma linha por P-BR com score, limiar, margem e fpr_exigido nos dois bracos, e o tipo: perdida, ganha ou igual."""
    saida = pbr.copy()
    for papel, braco in (("base", base), ("novo", novo)):
        c = chamadas[braco].reindex(saida.index)
        saida[f"score_{papel}"], saida[f"limiar_{papel}"] = c["score"].to_numpy(), c["limiar"].to_numpy()
        saida[f"margem_{papel}"] = saida[f"score_{papel}"] - saida[f"limiar_{papel}"]
        saida[f"positiva_{papel}"] = (c["chamada"] == "positive").to_numpy()
        saida[f"fpr_exigido_{papel}"] = [fpr_exigido(s, benignas[braco][r]) if np.isfinite(s) else np.nan
                                         for s, r in zip(saida[f"score_{papel}"], saida["run"])]
        saida[f"fpr_no_limiar_{papel}"] = [fpr_exigido(t, benignas[braco][r]) if np.isfinite(t) else np.nan
                                           for t, r in zip(saida[f"limiar_{papel}"], saida["run"])]
    perdida = saida["positiva_base"] & ~saida["positiva_novo"]
    ganha = saida["positiva_novo"] & ~saida["positiva_base"]
    saida["tipo"] = np.where(perdida, "perdida", np.where(ganha, "ganha", "igual"))
    # Mecanismo: a perda e de ponto de operacao se o novo ainda chamaria a variante no falso-positivo da base.
    saida["mecanismo"] = np.where(
        perdida, np.where(saida["fpr_exigido_novo"] <= saida["fpr_no_limiar_base"], "ponto_de_operacao", "ordenacao"),
        np.where(ganha, np.where(saida["fpr_exigido_base"] <= saida["fpr_no_limiar_novo"], "ponto_de_operacao",
                                 "ordenacao"), ""))
    saida["faixa_abraom"] = saida["abraom_af"].map(_faixa)
    saida["faixa_gnomad"] = (saida["gnomad_v4_af"].map(_faixa) if "gnomad_v4_af" in saida
                             else "sem coluna")
    saida["razao_abraom_gnomad"] = ([_razao(a, g) for a, g in zip(saida["abraom_af"], saida["gnomad_v4_af"])]
                                    if "gnomad_v4_af" in saida else "sem coluna")
    return saida


def resumo_do_grupo(linhas: pd.DataFrame) -> dict[str, Any]:
    return {"n": int(len(linhas)), "por_tier": _contagem(linhas["label_tier"]),
            "por_painel": _contagem(linhas["primary_panel"]), "por_execucao": _contagem(linhas["run"].astype(str)),
            "mecanismo": _contagem(linhas["mecanismo"]), "faixa_abraom": _contagem(linhas["faixa_abraom"]),
            "faixa_gnomad": _contagem(linhas["faixa_gnomad"]),
            "razao_abraom_gnomad": _contagem(pd.Series(linhas["razao_abraom_gnomad"])),
            "margem_base": _resumo_numerico(linhas["margem_base"]), "margem_novo": _resumo_numerico(linhas["margem_novo"]),
            "fpr_exigido_base": _resumo_numerico(linhas["fpr_exigido_base"]),
            "fpr_exigido_novo": _resumo_numerico(linhas["fpr_exigido_novo"])}


def especificidade_equivalente(pbr: pd.DataFrame, chamadas: dict[str, pd.DataFrame],
                               benignas: dict[str, dict[int, np.ndarray]], base: str, novo: str) -> dict[str, Any]:
    """Analise POSTERIOR: o limiar do novo refeito, por execucao, para a especificidade da base na validation."""
    por_run, limiares_novos = {}, {}
    for run in sorted(set(chamadas[base]["run"].astype(int))):
        limiar_base = float(chamadas[base].loc[chamadas[base]["run"] == run, "limiar"].iloc[0])
        alvo = 1.0 - fpr_exigido(limiar_base, benignas[base][run])
        limiares_novos[run] = limiar_com_especificidade(benignas[novo][run], alvo)
        por_run[str(run)] = {"especificidade_alvo": alvo, "limiar_original_do_novo":
                             float(chamadas[novo].loc[chamadas[novo]["run"] == run, "limiar"].iloc[0]),
                             "limiar_equivalente_do_novo": limiares_novos[run],
                             "especificidade_obtida": 1.0 - fpr_exigido(limiares_novos[run], benignas[novo][run])}
    c_base, c_novo = chamadas[base].reindex(pbr.index), chamadas[novo].reindex(pbr.index)
    positiva_base = (c_base["chamada"] == "positive").to_numpy()
    corte = pbr["run"].map(limiares_novos).to_numpy(dtype=float)
    positiva_novo = np.isfinite(c_novo["score"].to_numpy()) & (c_novo["score"].to_numpy() >= corte)
    return {"natureza": "posterior ao teste; limiares na validation; nao substitui o resultado do passo 4",
            "por_execucao": por_run, "sensibilidade_base": float(positiva_base.mean()),
            "sensibilidade_novo": float(positiva_novo.mean()),
            "perdidas": int((positiva_base & ~positiva_novo).sum()), "ganhas": int((positiva_novo & ~positiva_base).sum())}


def criticas_com_fpr(lista: list[dict[str, Any]], raiz: Path, chamadas: dict[str, pd.DataFrame],
                     benignas: dict[str, dict[int, np.ndarray]]) -> list[dict[str, Any]]:
    release = leituras.carregar_release(raiz)
    saida = []
    for item in criticas_do_release(lista, release):
        if not item["no_release"]:
            saida.append({**item, "bracos": None, "nota": "fora do release: sem score nesta fase"})
            continue
        vid, por_braco = item["variant_id"], {}
        for braco in bracos.BRACOS:
            if vid not in chamadas[braco].index:
                por_braco[braco] = {"chamada": None, "nota": "sem score"}
                continue
            c = chamadas[braco].loc[vid]
            por_braco[braco] = {"score": float(c["score"]), "limiar": float(c["limiar"]),
                                "margem": float(c["score"] - c["limiar"]), "chamada": str(c["chamada"]),
                                "fpr_exigido": fpr_exigido(float(c["score"]), benignas[braco][int(c["run"])]),
                                "fpr_no_limiar": fpr_exigido(float(c["limiar"]), benignas[braco][int(c["run"])])}
        saida.append({**item, "bracos": por_braco})
    return saida


# ---------------------------------------------------------------------------------------------- relatorio

def _f(valor: Any, casas: int = 3) -> str:
    return "—" if valor is None or (isinstance(valor, float) and not math.isfinite(valor)) else f"{valor:.{casas}f}"


def relatorio(d: dict[str, Any]) -> str:
    L = ["# Fase 1: diagnóstico das perdas e ganhos de P-BR (posterior ao teste)", "",
         "Desenvolvimento exploratório. Não altera o resultado do passo 4 nem a regra de segurança. `fpr exigido` = "
         "falso-positivo de validation que a chamada da variante exigiria; mecanismo `ponto_de_operacao` = o braço novo "
         "ainda chamaria a variante no falso-positivo de validation da base.", ""]
    for nome, par in d["pares"].items():
        perd, ganh, eq = par["perdidas"], par["ganhas"], par["especificidade_equivalente"]
        L += [f"## {nome}", "",
              f"- **Perdidas:** {perd['n']} | tier {perd['por_tier']} | mecanismo {perd['mecanismo']}",
              f"  - painel {perd['por_painel']}",
              f"  - ABraOM {perd['faixa_abraom']} | gnomAD {perd['faixa_gnomad']} | razão {perd['razao_abraom_gnomad']}",
              f"  - fpr exigido no novo: mediana {_f(perd['fpr_exigido_novo']['mediana'])} (p10 "
              f"{_f(perd['fpr_exigido_novo']['p10'])}; p90 {_f(perd['fpr_exigido_novo']['p90'])})",
              f"- **Ganhas:** {ganh['n']} | tier {ganh['por_tier']} | mecanismo {ganh['mecanismo']}",
              f"- **Especificidade equivalente (posterior):** sensibilidade base {_f(eq['sensibilidade_base'])}, "
              f"novo {_f(eq['sensibilidade_novo'])}; perdidas {eq['perdidas']}, ganhas {eq['ganhas']}", ""]
    L += ["## Críticas: fpr exigido por braço (chamada)", "", "| gene | variante | ABraOM | tier | "
          + " | ".join(bracos.BRACOS) + " |", "|---|---|---|---|" + "---|" * len(bracos.BRACOS)]
    for c in d["criticas"]:
        if c["bracos"] is None:
            L.append(f"| {c['gene']} | {c['hgvs']} | — | — | " + " | ".join("fora do release" for _ in bracos.BRACOS)
                     + " |")
            continue
        celulas = []
        for braco in bracos.BRACOS:
            b = c["bracos"][braco]
            celulas.append("sem score" if b.get("chamada") is None else
                           f"{_f(b['fpr_exigido'])} ({'+' if b['chamada'] == 'positive' else '−'})")
        L.append(f"| {c['gene']} | {c['hgvs']} | {'sim' if c['present_abraom'] else 'não'} | {c['tier']} | "
                 + " | ".join(celulas) + " |")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------------------------------- main

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--entrega", required=True, type=Path)
    parser.add_argument("--bracos", required=True, type=Path, help="pasta do passo 3")
    parser.add_argument("--leituras", required=True, type=Path, help="pasta do passo 4")
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    entrega, pasta_bracos = args.entrega.expanduser(), args.bracos.expanduser()
    pasta_leituras, destino = args.leituras.expanduser(), args.out_dir.expanduser()
    if destino.exists():
        return leituras._falhar([f"{destino} ja existe; o diagnostico grava sempre numa pasta nova"])
    raiz = entrega / bracos.RELEASE
    try:
        lidas = json.loads((pasta_leituras / "leituras.json").read_text(encoding="utf-8"))
        if Path(lidas["proveniencia"]["bracos"]).expanduser().resolve() != pasta_bracos.resolve():
            raise FalhaDaFase1(f"o passo 4 leu outro passo 3: {lidas['proveniencia']['bracos']}")
        fontes = json.loads((pasta_bracos / "fontes.json").read_text(encoding="utf-8"))
        leituras.conferir_fontes(fontes, raiz)
        preds = leituras.carregar_predicoes(pasta_bracos)
        limiares = {b: {int(r): float(t) for r, t in v.items()} for b, v in lidas["limiares"].items()}
        chamadas = {b: leituras.chamadas_de_teste(p, limiares[b]) for b, p in preds.items()}
        pbr = carregar_p_br(raiz)
        rotulo = leituras.carregar_release(raiz).set_index("variant_id")["binary_label"].astype(int)
        benignas = benignas_da_validacao(preds, rotulo)
        lista_de_criticas = ler_criticas(entrega / leituras.CRITICAS)
    except (FalhaDaFase1, ValueError, KeyError, OSError) as exc:
        return leituras._falhar([f"{type(exc).__name__}: {exc}"])

    pares, tabelas = {}, []
    for base, novo in leituras.PARES:
        nome = leituras.nome_do_par(base, novo)
        linhas = linhas_do_par(pbr, chamadas, benignas, base, novo)
        gravado = lidas["p_br"]["pares"][nome]
        for tipo, chave in (("perdida", "ids_perdidas"), ("ganha", "ids_ganhas")):
            if set(linhas.index[linhas["tipo"] == tipo]) != set(gravado[chave]):
                return leituras._falhar([f"{nome}: as {tipo}s nao reproduzem as do passo 4"])
        pares[nome] = {"base": base, "novo": novo,
                       "perdidas": resumo_do_grupo(linhas[linhas["tipo"] == "perdida"]),
                       "ganhas": resumo_do_grupo(linhas[linhas["tipo"] == "ganha"]),
                       "especificidade_equivalente": especificidade_equivalente(pbr, chamadas, benignas, base, novo)}
        tabelas.append(linhas[linhas["tipo"] != "igual"].assign(par=nome))
    diagnostico = {
        "formato": "fase1_diagnostico_v1",
        "natureza": "desenvolvimento exploratorio, definido depois do teste; nao altera o passo 4",
        "fontes": {"bracos": str(pasta_bracos), "leituras": str(pasta_leituras),
                   "leituras_sha256": sha256_do_arquivo(pasta_leituras / "leituras.json"),
                   "script_sha256": sha256_do_arquivo(Path(__file__)), "revisao": bracos.revisao_do_repositorio()},
        "n_p_br": int(len(pbr)), "pares": pares,
        "criticas": criticas_com_fpr(lista_de_criticas, raiz, chamadas, benignas),
    }
    destino.mkdir(parents=True)
    (destino / "diagnostico.json").write_text(json.dumps(diagnostico, indent=2, ensure_ascii=False,
                                                         default=leituras._json), encoding="utf-8")
    texto = relatorio(json.loads((destino / "diagnostico.json").read_text(encoding="utf-8")))
    (destino / "diagnostico.md").write_text(texto, encoding="utf-8")
    pd.concat(tabelas, ignore_index=True).drop(columns=["core_purged_runs"], errors="ignore").to_csv(
        destino / "perdas_e_ganhos.csv", index=False)
    leituras._imprimir(texto)
    print(f"PASSOU: diagnostico em {destino}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
