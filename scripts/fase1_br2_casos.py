#!/usr/bin/env python3
"""Fase 1, BR v2: as P-BR perdidas e recuperadas, uma a uma (desenvolvimento exploratorio, posterior ao teste).

Pedido da revisao de 06/10 (docs/fase1_br2_resultado.md, proximos passos). Nao muda resultados nem regras: so le as
predicoes e os limiares ja gravados e o release. No limiar MCC original de cada braco, separa:
- RECUPERADAS: perdidas por E+F -> E+F+BR e nao por E+F -> E+F+BR2;
- MANTIDAS: perdidas pelos dois;
- NOVAS: perdidas so por E+F -> E+F+BR2;
- e, so na contagem e no CSV, as GANHAS pelo BR2 (positivas no BR2 e nao em E+F).
Os conjuntos tem de reproduzir os do br2.json. Para cada variante acrescenta:
- o SINAL FUNCIONAL: score, chamada e fpr exigido de E sozinho (passo 3, limiar do passo 4). E a pergunta da cabeca
  com interacao: o sinal funcional aponta para patogenicidade nas perdas?
- o SUPORTE do ABraOM: AC/AN, FILTER, fracao de AN e o limite inferior unilateral de 95% de Clopper-Pearson de AC/AN
  (o criterio de `mosaic.regional_truth.af_lower_bound`), ao lado da AF, da popmax e da fafmax do gnomAD. A comparacao
  do limite inferior com o gnomAD e descritiva (nao testa diferenca) e separa valor ausente de frequencia observada;
- em E+F, E+F+BR e E+F+BR2: score, limiar, margem, chamada e fpr exigido.
E e o classificador so com o embedding: uma previsao aprendida, nao evidencia funcional experimental. Nada aqui diz que
uma variante e fundadora ou mais frequente no Brasil: e a tabela para olhar caso a caso.

USO (notebook, no .venv do Mosaic; leva poucos minutos)
    PYTHONPATH="$PWD" uv run --project ~/mosaic-v1-2026-09-30 --frozen python scripts/fase1_br2_casos.py \\
        --entrega ~/mosaic-v1-2026-09-30 --bracos <passo 3> --leituras <passo 4> --br2 <treino do BR v2> \\
        --leituras-br2 <leituras do BR v2> --out-dir <nova>

SAIDAS (em --out-dir, que nao pode existir): casos.json, casos.md e casos.csv.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from eval.campanha.cache import sha256_do_arquivo  # noqa: E402
from scripts import fase1_bracos as bracos  # noqa: E402
from scripts import fase1_diagnostico as diag  # noqa: E402
from scripts import fase1_leituras as leituras  # noqa: E402
from scripts.fase1_br2_ler import NOVO, ORIGINAL  # noqa: E402
from scripts.fase1_br2_treinar import AN_MAXIMO, AN_SUFICIENTE, BANDAS, BRACO, ID  # noqa: E402
from scripts.inventario_mosaic_v1 import K  # noqa: E402

FalhaDaFase1 = bracos.FalhaDaFase1
BRACOS = ("E", "E+F", "E+F+BR", BRACO)
SLUG = {"E": "e", "E+F": "e_f", "E+F+BR": "e_f_br", BRACO: "e_f_br2"}
GRUPOS = ("mantida", "nova", "recuperada", "ganha_br2")
#: Os tres grupos que a revisao pediu olhar um a um; as ganhas ficam so na contagem e no CSV.
NA_TABELA = ("mantida", "nova", "recuperada")


# ------------------------------------------------------------------------------------------------- quantidades

def limite_inferior(ac: Any, an: Any) -> float:
    """Limite inferior unilateral de 95% de Clopper-Pearson de AC/AN: o p com P(X >= ac | an, p) = 0,05, ou seja,
    P(X <= ac - 1) = 0,95, pela bissecao de `limite_superior_cp`. Zero sem copias; NaN sem AN."""
    if ac is None or an is None or pd.isna(ac) or pd.isna(an) or an <= 0:
        return math.nan
    x, n = int(ac), int(an)
    if x <= 0:
        return 0.0
    return leituras.limite_superior_cp(x - 1, n, confianca=0.05)


def faixa_de_ac(ac: Any) -> str:
    if ac is None or pd.isna(ac) or ac < 1:
        return "sem copias"
    return next(nome for baixo, alto, nome in BANDAS if baixo <= ac <= (alto if alto is not None else math.inf))


def grupos(positivas: dict[str, pd.Series]) -> dict[str, list[str]]:
    """Os grupos no limiar MCC: perdas de E+F -> E+F+BR e de E+F -> E+F+BR2 e as ganhas pelo BR2."""
    base, br, br2 = positivas["E+F"], positivas["E+F+BR"], positivas[BRACO]
    perdida_br, perdida_br2 = base & ~br, base & ~br2
    conjuntos = {"mantida": perdida_br & perdida_br2, "nova": perdida_br2 & ~perdida_br,
                 "recuperada": perdida_br & ~perdida_br2, "ganha_br2": br2 & ~base}
    return {g: sorted(m.index[m].astype(str)) for g, m in conjuntos.items()}


def conferir_grupos(g: dict[str, list[str]], br2: dict[str, Any]) -> None:
    """Os grupos recontados tem de ser os das leituras do BR v2 (perdas e ganhos gravados no br2.json)."""
    pares = br2["p_br"]["pares"]
    esperado = {"perdidas_br": set(pares[ORIGINAL]["ids_perdidas"]), "perdidas_br2": set(pares[NOVO]["ids_perdidas"]),
                "ganhas_br2": set(pares[NOVO]["ids_ganhas"])}
    recontado = {"perdidas_br": set(g["mantida"]) | set(g["recuperada"]),
                 "perdidas_br2": set(g["mantida"]) | set(g["nova"]), "ganhas_br2": set(g["ganha_br2"])}
    for chave in esperado:
        if esperado[chave] != recontado[chave]:
            raise FalhaDaFase1(f"{chave}: os grupos recontados nao sao os do br2.json")


def _coluna(frame: pd.DataFrame, nome: str) -> pd.Series:
    return frame[nome] if nome in frame else pd.Series(np.nan, index=frame.index)


def comparar(limite: np.ndarray, referencia: pd.Series) -> list[str]:
    """`acima`, `nao_acima` ou `sem_valor`: um valor ausente no gnomAD (ou sem limite) nao vira zero."""
    ref = pd.to_numeric(referencia, errors="coerce").to_numpy(dtype=float)
    return ["sem_valor" if not (np.isfinite(li) and np.isfinite(r)) else "acima" if li > r else "nao_acima"
            for li, r in zip(limite, ref)]


def tabela(pbr: pd.DataFrame, g: dict[str, list[str]], chamadas: dict[str, pd.DataFrame],
           benignas: dict[str, dict[int, np.ndarray]]) -> pd.DataFrame:
    """Uma linha por variante dos grupos, com o sinal funcional, o suporte do ABraOM e as chamadas dos quatro bracos."""
    ordem = [(grupo, vid) for grupo in GRUPOS for vid in g[grupo]]
    ids = [vid for _, vid in ordem]
    linhas = pbr.reindex(ids)
    gene = _coluna(linhas, "mane_gene").fillna(_coluna(linhas, "clinvar_gene_symbol"))
    ac, an = _coluna(linhas, "abraom_ac"), _coluna(linhas, "abraom_an")
    li = np.array([limite_inferior(a, n) for a, n in zip(ac, an)], dtype=float)
    af_g = pd.to_numeric(_coluna(linhas, "gnomad_v4_af"), errors="coerce")
    popmax = pd.to_numeric(_coluna(linhas, "gnomad_v4_popmax_af"), errors="coerce")
    fafmax = pd.to_numeric(_coluna(linhas, "gnomad_v4_fafmax_faf95"), errors="coerce")
    saida = pd.DataFrame({
        "grupo": [grupo for grupo, _ in ordem], "variant_id": ids, "run": _coluna(linhas, "run").to_numpy(),
        "label_tier": _coluna(linhas, "label_tier").to_numpy(), "primary_panel": _coluna(linhas, "primary_panel").to_numpy(),
        "gene": gene.to_numpy(), "consequence": _coluna(linhas, "consequence").to_numpy(),
        "aa_change": _coluna(linhas, "aa_change").to_numpy(),
        "clinvar_classificacao": _coluna(linhas, "aggregate_classification").to_numpy(),
        "clinvar_revisao": _coluna(linhas, "aggregate_review_status").to_numpy(),
        "br_lab_any": _coluna(linhas, "br_lab_any").to_numpy(),
        "abraom_ac": ac.to_numpy(), "abraom_an": an.to_numpy(),
        "abraom_fracao_de_an": (pd.to_numeric(an, errors="coerce") / AN_MAXIMO).to_numpy(),
        "abraom_an_suficiente": (pd.to_numeric(an, errors="coerce") >= AN_SUFICIENTE).to_numpy(),
        "abraom_filter": _coluna(linhas, "abraom_filter").to_numpy(),
        "abraom_af": _coluna(linhas, "abraom_af").to_numpy(), "abraom_li95": li,
        "faixa_de_ac": [faixa_de_ac(a) for a in ac],
        "gnomad_status": _coluna(linhas, "gnomad_status").to_numpy(), "gnomad_af": af_g.to_numpy(),
        "gnomad_popmax_af": popmax.to_numpy(),
        "gnomad_faf95": pd.to_numeric(_coluna(linhas, "gnomad_v4_faf95"), errors="coerce").to_numpy(),
        "gnomad_fafmax": fafmax.to_numpy(),
        "gnomad_af_amr": pd.to_numeric(_coluna(linhas, "gnomad_v4_af_amr"), errors="coerce").to_numpy(),
        "gnomad_filter": _coluna(linhas, "gnomad_v4_filter").to_numpy(),
        "li95_vs_af_gnomad": comparar(li, af_g), "li95_vs_fafmax": comparar(li, fafmax),
        "phylop_241way": _coluna(linhas, "phylop_241way").to_numpy()})
    for braco in BRACOS:
        c = chamadas[braco].reindex(ids)
        s, t = c["score"].to_numpy(dtype=float), c["limiar"].to_numpy(dtype=float)
        runs = saida["run"].astype(int).to_numpy()
        p = SLUG[braco]
        saida[f"score_{p}"], saida[f"limiar_{p}"], saida[f"margem_{p}"] = s, t, s - t
        saida[f"positiva_{p}"] = (c["chamada"] == "positive").to_numpy()
        saida[f"fpr_exigido_{p}"] = [diag.fpr_exigido(v, benignas[braco][r]) if np.isfinite(v) else np.nan
                                     for v, r in zip(s, runs)]
        saida[f"fpr_no_limiar_{p}"] = [diag.fpr_exigido(v, benignas[braco][r]) for v, r in zip(t, runs)]
    return saida


def _mediana(valores: pd.Series) -> float | None:
    v = pd.to_numeric(valores, errors="coerce").dropna()
    return float(v.median()) if len(v) else None


def resumo(t: pd.DataFrame) -> dict[str, Any]:
    """Contagens por grupo: tier, faixa de AC, qualidade do ABraOM, suporte contra o gnomAD e o sinal funcional."""
    saida = {}
    for grupo in GRUPOS:
        g = t[t["grupo"] == grupo]
        saida[grupo] = {
            "n": int(len(g)), "gold": int((g["label_tier"] == "gold").sum()),
            "por_painel": diag._contagem(g["primary_panel"]), "faixa_de_ac": diag._contagem(g["faixa_de_ac"]),
            "abraom_pass": int((g["abraom_filter"] == "PASS").sum()),
            "abraom_an_suficiente": int(g["abraom_an_suficiente"].sum()),
            "li95_vs_af_gnomad": diag._contagem(g["li95_vs_af_gnomad"]),
            "li95_vs_fafmax": diag._contagem(g["li95_vs_fafmax"]),
            "gnomad_status": diag._contagem(g["gnomad_status"]),
            "e_positiva": int(g["positiva_e"].sum()),
            "mediana_fpr_exigido_e": _mediana(g["fpr_exigido_e"]),
            "mediana_margem_e_f": _mediana(g["margem_e_f"]),
            "clinvar_revisao": diag._contagem(g["clinvar_revisao"])}
    return saida


# ----------------------------------------------------------------------------------------------- relatorio

def _f(valor: Any, casas: int = 3) -> str:
    return "—" if valor is None or (isinstance(valor, float) and not math.isfinite(valor)) else f"{valor:.{casas}f}"


def _af(valor: Any) -> str:
    return "—" if valor is None or pd.isna(valor) else f"{float(valor):.2e}"


def _texto(*valores: Any) -> str:
    """O primeiro valor de texto nao vazio, ou um travessao."""
    return next((str(v) for v in valores if isinstance(v, str) and v), "—")


def _comparacao(contagem: dict[str, int]) -> str:
    """`acima/com valor`, e os sem valor a parte."""
    acima, com_valor = contagem.get("acima", 0), contagem.get("acima", 0) + contagem.get("nao_acima", 0)
    sem = contagem.get("sem_valor", 0)
    return f"{acima}/{com_valor}" + (f" ({sem} sem valor)" if sem else "")


def _chamada(linha: pd.Series, p: str) -> str:
    return f"{_f(linha[f'fpr_exigido_{p}'])} ({'+' if linha[f'positiva_{p}'] else '−'})"


def relatorio(d: dict[str, Any], t: pd.DataFrame) -> str:
    r = d["resumo"]
    L = ["# BR v2: P-BR perdidas e recuperadas, caso a caso (posterior ao teste)", "",
         "Desenvolvimento exploratório; não altera resultados nem regras. Grupos no limiar MCC original: **mantida** = "
         "perdida por E+F+BR e por E+F+BR2; **nova** = só por E+F+BR2; **recuperada** = só por E+F+BR. Em cada braço, "
         "`fpr exigido (chamada)`: o falso-positivo de validation que a chamada exigiria. **E** é o classificador só "
         "com o embedding: previsão aprendida, não evidência funcional experimental. LI 95% = limite inferior "
         "unilateral de Clopper-Pearson de AC/AN do ABraOM. A comparação do LI 95% com o gnomAD é descritiva: não "
         "testa diferença nem demonstra enriquecimento; valor ausente no gnomAD fica separado (`sem valor`), não vira "
         "zero.", "",
         "## Resumo por grupo", "",
         "| grupo | n | gold | E positiva | AC 1–2 | AC ≥ 3 | PASS | AN ≥ 80% | LI 95% > AF gnomAD | LI 95% > fafmax "
         "| mediana fpr exigido E | mediana margem E+F |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for grupo in GRUPOS:
        g = r[grupo]
        faixas = g["faixa_de_ac"]
        poucas = faixas.get("ac1", 0) + faixas.get("ac2", 0)
        muitas = faixas.get("ac3a9", 0) + faixas.get("ac10mais", 0)
        L.append(f"| {grupo} | {g['n']} | {g['gold']} | {g['e_positiva']} | {poucas} | {muitas} | {g['abraom_pass']} | "
                 f"{g['abraom_an_suficiente']} | {_comparacao(g['li95_vs_af_gnomad'])} | "
                 f"{_comparacao(g['li95_vs_fafmax'])} | "
                 f"{_f(g['mediana_fpr_exigido_e'])} | {_f(g['mediana_margem_e_f'])} |")
    L += ["", "## Variantes (gold primeiro em cada grupo)", "",
          "| grupo | tier | gene | variante | painel | AC/AN | FILTER | AF ABraOM [LI 95%] | gnomAD AF · popmax · fafmax | "
          "E | E+F | E+F+BR | E+F+BR2 |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    vistas = t[t["grupo"].isin(NA_TABELA)].assign(
        _ordem_grupo=lambda x: x["grupo"].map({g: i for i, g in enumerate(NA_TABELA)}),
        _ordem_tier=lambda x: (x["label_tier"] != "gold").astype(int))
    for _, linha in vistas.sort_values(["_ordem_grupo", "_ordem_tier", "gene"], kind="stable").iterrows():
        acan = ("—" if pd.isna(linha["abraom_ac"]) else f"{int(linha['abraom_ac'])}/{int(linha['abraom_an'])}")
        L.append(f"| {linha['grupo']} | {linha['label_tier']} | {_texto(linha['gene'])} | "
                 f"{_texto(linha['aa_change'], linha['consequence'])} | {linha['primary_panel']} | "
                 f"{acan} | {_texto(linha['abraom_filter'])} | {_af(linha['abraom_af'])} [{_af(linha['abraom_li95'])}] | "
                 f"{_af(linha['gnomad_af'])} · {_af(linha['gnomad_popmax_af'])} · {_af(linha['gnomad_fafmax'])} | "
                 + " | ".join(_chamada(linha, SLUG[b]) for b in BRACOS) + " |")
    L += ["", f"Limite inferior conferido com o `af_lower_bound` do Mosaic: {d['limite_inferior_conferido_com_mosaic']}.",
          "Não se conclui aqui que uma variante seja fundadora ou mais frequente no Brasil."]
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------------------------------- main

def _pastas_do_br2(proveniencia: str) -> tuple[Path, Path]:
    m = re.fullmatch(r"(.*) \(E\+F, E\+F\+BR\) e (.*) \(" + re.escape(BRACO) + r"\)", proveniencia)
    if m is None:
        raise FalhaDaFase1(f"proveniencia das leituras do BR v2 inesperada: {proveniencia}")
    return Path(m.group(1)).expanduser().resolve(), Path(m.group(2)).expanduser().resolve()


def conferir_com_o_mosaic(t: pd.DataFrame) -> bool | None:
    """Com o pacote mosaic (notebook), o limite inferior tem de bater com o `af_lower_bound`; sem ele, None."""
    try:
        from mosaic.regional_truth import af_lower_bound
    except ImportError:
        return None
    m = t["abraom_ac"].notna() & t["abraom_an"].notna() & (pd.to_numeric(t["abraom_ac"], errors="coerce") > 0)
    if not m.any():
        return True
    oficial = np.asarray(af_lower_bound(t.loc[m, "abraom_ac"].to_numpy(dtype=float),
                                        t.loc[m, "abraom_an"].to_numpy(dtype=float)), dtype=float)
    if not np.allclose(oficial, t.loc[m, "abraom_li95"].to_numpy(dtype=float), rtol=1e-6, atol=1e-12):
        raise FalhaDaFase1("o limite inferior nao bate com o af_lower_bound do Mosaic")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--entrega", required=True, type=Path)
    parser.add_argument("--bracos", required=True, type=Path, help="pasta do passo 3")
    parser.add_argument("--leituras", required=True, type=Path, help="pasta do passo 4 (limiar de E)")
    parser.add_argument("--br2", required=True, type=Path, help="pasta do treino do BR v2")
    parser.add_argument("--leituras-br2", required=True, type=Path, help="pasta das leituras do BR v2 (br2.json)")
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    entrega, pasta3, pasta4 = args.entrega.expanduser(), args.bracos.expanduser(), args.leituras.expanduser()
    pasta_br2, pasta_l2, destino = args.br2.expanduser(), args.leituras_br2.expanduser(), args.out_dir.expanduser()
    if destino.exists():
        return leituras._falhar([f"{destino} ja existe; o caso a caso grava sempre numa pasta nova"])
    raiz = entrega / bracos.RELEASE
    try:
        lidas4 = json.loads((pasta4 / "leituras.json").read_text(encoding="utf-8"))
        br2 = json.loads((pasta_l2 / "br2.json").read_text(encoding="utf-8"))
        if Path(lidas4["proveniencia"]["bracos"]).expanduser().resolve() != pasta3.resolve():
            raise FalhaDaFase1(f"o passo 4 leu outro passo 3: {lidas4['proveniencia']['bracos']}")
        if _pastas_do_br2(br2["proveniencia"]["bracos"]) != (pasta3.resolve(), pasta_br2.resolve()):
            raise FalhaDaFase1(f"as leituras do BR v2 leram outras pastas: {br2['proveniencia']['bracos']}")
        for pasta in (pasta3, pasta_br2):
            leituras.conferir_fontes(json.loads((pasta / "fontes.json").read_text(encoding="utf-8")), raiz)
        tres = leituras.carregar_predicoes(pasta3)
        novo = pd.read_parquet(pasta_br2 / ID / "predictions.parquet")
        preds = {"E": tres["E"], "E+F": tres["E+F"], "E+F+BR": tres["E+F+BR"],
                 BRACO: novo.assign(variant_id=novo["variant_id"].astype(str), run=novo["run"].astype(int))}
        limiares = {b: {int(r): float(v) for r, v in br2["limiares"][b].items()} for b in ("E+F", "E+F+BR", BRACO)}
        limiares["E"] = {int(r): float(v) for r, v in lidas4["limiares"]["E"].items()}
        for b in ("E+F", "E+F+BR"):
            if any(not np.isclose(float(lidas4["limiares"][b][str(r)]), limiares[b][r], rtol=0, atol=1e-12)
                   for r in range(K)):
                raise FalhaDaFase1(f"{b}: limiares do br2.json diferentes dos do passo 4")
        chamadas = {b: leituras.chamadas_de_teste(p, limiares[b]) for b, p in preds.items()}
        rotulo = leituras.carregar_release(raiz).set_index("variant_id")["binary_label"].astype(int)
        benignas = diag.benignas_da_validacao(preds, rotulo)
        pbr = diag.carregar_p_br(raiz)
        positivas = {b: chamadas[b]["chamada"].reindex(pbr.index).eq("positive") for b in ("E+F", "E+F+BR", BRACO)}
        g = grupos(positivas)
        conferir_grupos(g, br2)
        t = tabela(pbr, g, chamadas, benignas)
        conferido = conferir_com_o_mosaic(t)
    except (FalhaDaFase1, ValueError, KeyError, OSError) as exc:
        return leituras._falhar([f"{type(exc).__name__}: {exc}"])

    d = {"formato": "fase1_br2_casos_v1",
         "natureza": "desenvolvimento exploratorio posterior ao teste; descritivo; nao altera resultados nem regras",
         "grupos": {k: len(v) for k, v in g.items()}, "ids": g, "resumo": resumo(t),
         "limite_inferior_conferido_com_mosaic": conferido,
         "fontes": {"bracos": str(pasta3), "leituras": str(pasta4), "br2": str(pasta_br2), "leituras_br2": str(pasta_l2),
                    "br2_json_sha256": sha256_do_arquivo(pasta_l2 / "br2.json"),
                    "script_sha256": sha256_do_arquivo(Path(__file__)), "revisao": bracos.revisao_do_repositorio()}}
    destino.mkdir(parents=True)
    (destino / "casos.json").write_text(json.dumps(d, indent=2, ensure_ascii=False, default=leituras._json),
                                        encoding="utf-8")
    t.to_csv(destino / "casos.csv", index=False)
    texto = relatorio(json.loads((destino / "casos.json").read_text(encoding="utf-8")), t)
    (destino / "casos.md").write_text(texto, encoding="utf-8")
    leituras._imprimir(texto)
    print(f"PASSOU: caso a caso em {destino}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
