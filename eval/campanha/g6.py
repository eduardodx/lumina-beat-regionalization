"""G6: o sistema que vai ao G7, conferido e congelado. Sem torch: roda no Windows.

O que se congela (secao `g6` de `configs/campanha_r03_desenvolvimento.json`):
    - a composicao final: M0 = h11, h12, h13; MR = a1+h11, a2+h12, a3+h13. Nenhuma troca, nenhuma mistura das nove
      cabecas MR dos tres comparadores, nenhuma escolha depois de ver resultado;
    - a predicao do sistema: a media das tres probabilidades calibradas;
    - o limiar do ensemble: a regra do protocolo (`calibrate_threshold` do Mosaic) nessa media, no fold 1, por
      sistema;
    - margens, bootstrap, analises secundarias e proveniencia, como declarados. Sem margens, sem a unidade do
      bootstrap e com pendencia aberta o manifesto nao congela.

Aqui ficam as regras puras: o limiar, a conferencia da composicao contra as conferencias das cabecas, a comparacao
descritiva do ensemble final, os bloqueios do congelamento, a montagem e a gravacao do manifesto (sha256 dos bytes em
arquivo a parte) e a leitura de um manifesto congelado. Recarregar e pontuar as cabecas (torch) fica em
`scripts/construir_g6.py`.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from eval.campanha import metricas
from eval.campanha.estudos import com_limiar

FORMATO_DO_MANIFESTO = "campanha_r03_g6_manifesto_v1"
NOME_DO_MANIFESTO = "g6_manifesto.json"
NOME_DO_RASCUNHO = "g6_manifesto_rascunho.json"
CONGELADO, RASCUNHO = "CONGELADO", "RASCUNHO"
SISTEMAS = ("M0", "MR")
#: Nome de cada sistema no Mosaic (`BRAZIL_SYSTEMS`) e no consumidor (`estudos.BASE`, `estudos.REGIONALIZADO`).
NO_MOSAIC = {"M0": "base", "MR": "regionalized"}
REGRA_DE_PREDICAO = "score >= threshold"
REGRA_DO_LIMIAR = ("calibrate_threshold do Mosaic (814e7f0; suite.yaml threshold: metric mcc, on validation_gold, "
                   "tiebreak [specificity, higher_threshold]): candidatos logo abaixo do menor score, cada score "
                   "distinto e logo acima do maior; maior MCC, empate -> maior especificidade -> maior limiar")
ESTADOS_RESOLVIDOS = ("FEITO", "RETIRADO")
#: Codigo que produz o G6 e codigo que o G7 roda: o sha256 de cada um vai no manifesto.
CODIGO_DO_G6 = ("eval/campanha/g6.py", "scripts/construir_g6.py", "eval/campanha/cabeca.py",
                "eval/campanha/metricas.py", "eval/campanha/leitura_do_cache.py", "eval/campanha/recortes.py")
CODIGO_DO_G7 = ("eval/campanha/estudos.py", "eval/campanha/metricas.py", "eval/campanha/cabeca.py",
                "eval/campanha/g6.py", "eval/campanha/g7.py", "eval/campanha/baselines.py",
                "scripts/extrair_estudos.py", "scripts/avaliar_estudos.py",
                "scripts/conferir_cobertura_das_baselines.py")
#: Entradas das analises secundarias que o G7 le: o sha256 de cada uma vai no manifesto, e o G7 confere.
ENTRADAS_DAS_ANALISES_SECUNDARIAS = {
    "regra_ampla": "g2_regra_ampla/broad_brazilian_variant_ids.txt: controles com SCV de instituicao brasileira",
    "exposicao": "exposicao_por_membro.parquet do snapshot final (janela2048), raio 4.096 (n_janela por membro)",
}


class ManifestoInvalido(ValueError):
    """O manifesto nao confere com o proprio sha256, nao esta congelado ou nao e deste formato."""


# ----------------------------------------------------------------------------------------------- limiar

def candidatos_de_limiar(scores: Sequence[float]) -> np.ndarray:
    """Os do Mosaic (`threshold_candidates`): logo abaixo do menor, cada valor distinto e logo acima do maior."""
    s = np.asarray(scores, dtype=np.float64)
    s = s[np.isfinite(s)]
    if not s.size:
        return np.asarray([], dtype=np.float64)
    distintos = np.unique(s)
    return np.concatenate(([np.nextafter(distintos[0], -np.inf)], distintos, [np.nextafter(distintos[-1], np.inf)]))


def limiar_do_mosaic(rotulos: Sequence[int], scores: Sequence[float]) -> dict[str, Any]:
    """O `calibrate_threshold` do Mosaic com a mesma aritmetica: MCC 0 quando o denominador e 0 e a chave
    (MCC, especificidade, limiar) maximizada nessa ordem. Recusa o que o Mosaic devolveria como status de erro: aqui
    o conjunto e o fold 1 da campanha, que tem de ter as duas classes com score."""
    y = np.asarray(rotulos, dtype=int)
    s = np.asarray(scores, dtype=np.float64)
    if y.shape != s.shape:
        raise ValueError(f"rotulos {y.shape} e scores {s.shape} com formas diferentes")
    pontuados = np.isfinite(s)
    y_s, s_s = y[pontuados], s[pontuados]
    if not ((y_s == 1).any() and (y_s == 0).any()):
        raise ValueError("o limiar precisa das duas classes com score")
    melhor: tuple[tuple[float, float, float], float] | None = None
    for limiar in candidatos_de_limiar(s_s):
        previsto = s_s >= limiar
        vp, vn = int(((y_s == 1) & previsto).sum()), int(((y_s == 0) & ~previsto).sum())
        fp, fn = int(((y_s == 0) & previsto).sum()), int(((y_s == 1) & ~previsto).sum())
        denominador = (vp + fp) * (vp + fn) * (vn + fp) * (vn + fn)
        mcc = (vp * vn - fp * fn) / float(denominador) ** 0.5 if denominador else 0.0
        chave = (mcc, vn / (vn + fp), float(limiar))
        if melhor is None or chave > melhor[0]:
            melhor = (chave, vp / (vp + fn))
    (mcc, especificidade, limiar), sensibilidade = melhor
    return {"threshold": limiar, "mcc": mcc, "specificity": especificidade, "sensitivity": sensibilidade,
            "status": "ok", "prediction_rule": REGRA_DE_PREDICAO, "regra": REGRA_DO_LIMIAR,
            "n_total": int(y.size), "n_P_total": int((y == 1).sum()), "n_B_total": int((y == 0).sum()),
            "n_scored": int(pontuados.sum()), "n_P_scored": int((y_s == 1).sum()),
            "n_B_scored": int((y_s == 0).sum()), "coverage": float(pontuados.mean()) if y.size else None}


def media_dos_componentes(probabilidades: Sequence[np.ndarray]) -> np.ndarray:
    """A predicao do sistema: media simples das probabilidades calibradas, na ordem da composicao."""
    matriz = np.vstack([np.asarray(p, dtype=np.float64) for p in probabilidades])
    if matriz.shape[0] != 3:
        raise ValueError(f"o ensemble declarado tem 3 componentes, nao {matriz.shape[0]}")
    return matriz.mean(axis=0)


def sha256_dos_ids(ids: Iterable[Any]) -> str:
    """sha256 dos `variant_id` ordenados, um por linha: identifica o conjunto sem depender da ordem."""
    return hashlib.sha256("\n".join(sorted(map(str, ids))).encode("utf-8")).hexdigest()


# ----------------------------------------------------------------------------------------------- composicao

def componentes_declarados(campanha: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """A composicao final, conferida contra o pareamento de sementes, com o comparador e a coluna de cada
    componente. Recusa qualquer desvio: nao ha composicao alternativa a escolher."""
    composicao = campanha["g6"]["composicao_final"]
    pares = [(int(c["adapter"]), int(c["cabeca"])) for c in campanha["sementes"]["combinacoes"]]
    if [(int(c["adapter"]), int(c["cabeca"])) for c in composicao["MR"]] != pares:
        raise ValueError("a composicao do MR nao e o pareamento declarado em sementes.combinacoes")
    if [int(c["cabeca"]) for c in composicao["M0"]] != [h for _, h in pares] or \
            any("adapter" in c for c in composicao["M0"]):
        raise ValueError("a composicao do M0 nao e h_1, h_2, h_3 sem adapter")
    saida: dict[str, list[dict[str, Any]]] = {}
    for sistema in SISTEMAS:
        saida[sistema] = []
        for c in composicao[sistema]:
            partes = Path(c["arquivo"]).parts
            if len(partes) != 2 or partes[1] != f"cabeca_{sistema}_h{c['cabeca']}.pt":
                raise ValueError(f"{c['arquivo']}: esperado <comparador>/cabeca_{sistema}_h{c['cabeca']}.pt")
            adapter = int(c["adapter"]) if sistema == "MR" else None
            saida[sistema].append({
                "sistema": sistema, "cabeca": int(c["cabeca"]), "adapter": adapter, "arquivo": c["arquivo"],
                "comparador": partes[0], "coluna": f"{sistema}_h{c['cabeca']}",
                "rotulo": f"M0_h{c['cabeca']}" if sistema == "M0" else f"MR_a{adapter}_h{c['cabeca']}"})
        if len({c["arquivo"] for c in saida[sistema]}) != 3:
            raise ValueError(f"{sistema}: a composicao precisa de 3 arquivos distintos")
    if len({c["comparador"] for c in saida["M0"]}) != 1:
        raise ValueError("as cabecas do M0 tem de vir de um so comparador")
    return saida


def conferir_conferencias(conferencias: dict[str, dict[str, Any]], componentes: dict[str, list[dict[str, Any]]],
                          sha_dos_arquivos: dict[str, str]) -> list[str]:
    """Cada componente tem de ser o arquivo que a conferencia do seu comparador aprovou (mesmo sha256), e o M0 dos
    outros comparadores tem de ter sido conferido identico ao do comparador do M0. Pura."""
    problemas: list[str] = []
    referencia = componentes["M0"][0]["comparador"]
    nomes_do_m0 = sorted(Path(c["arquivo"]).name for c in componentes["M0"])
    for nome in sorted({c["comparador"] for s in SISTEMAS for c in componentes[s]}):
        conferencia = conferencias.get(nome)
        if conferencia is None:
            problemas.append(f"{nome}: sem conferencia_das_cabecas.json")
            continue
        if conferencia.get("passou") is not True or conferencia.get("problemas"):
            problemas.append(f"{nome}: a conferencia das cabecas nao passou")
        if nome == referencia:
            continue
        m0 = conferencia.get("m0_de_referencia") or {}
        if Path(str(m0.get("pasta", ""))).name != referencia:
            problemas.append(f"{nome}: o M0 nao foi conferido contra {referencia} (--m0-de-referencia)")
        cabecas = m0.get("cabecas") or {}
        if sorted(cabecas) != nomes_do_m0 or any(v != "identica" for v in cabecas.values()):
            problemas.append(f"{nome}: as cabecas do M0 nao foram conferidas identicas as de {referencia}: {cabecas}")
    for sistema in SISTEMAS:
        for c in componentes[sistema]:
            conferencia = conferencias.get(c["comparador"]) or {}
            if sistema == "MR" and conferencia.get("semente_do_adapter") != c["adapter"]:
                problemas.append(f"{c['arquivo']}: a conferencia de {c['comparador']} e da semente de adapter "
                                 f"{conferencia.get('semente_do_adapter')}, nao {c['adapter']}")
            linha = next((x for x in conferencia.get("cabecas", []) if x.get("arquivo") == Path(c["arquivo"]).name),
                         None)
            if linha is None:
                problemas.append(f"{c['arquivo']}: ausente da conferencia de {c['comparador']}")
            elif linha.get("sha256") != sha_dos_arquivos.get(c["arquivo"]):
                problemas.append(f"{c['arquivo']}: sha256 {str(sha_dos_arquivos.get(c['arquivo']))[:12]} != o da "
                                 f"conferencia {str(linha.get('sha256'))[:12]}")
            elif linha.get("problemas"):
                problemas.append(f"{c['arquivo']}: a conferencia registrou problemas: {linha['problemas']}")
    return problemas


def diferencas_numericas(recalculado: dict[str, Any], registrado: dict[str, Any], chaves: Iterable[str], *,
                         tolerancia: float, rotulo: str) -> list[str]:
    """Chaves em que o valor recalculado e o registrado diferem mais que a tolerancia (ou um falta). Pura."""
    problemas = []
    for chave in chaves:
        a, b = recalculado.get(chave), registrado.get(chave)
        if (a is None) != (b is None) or (a is not None and not abs(float(a) - float(b)) <= tolerancia):
            problemas.append(f"{rotulo} {chave}: recalculado {a} != registrado {b}")
    return problemas


# ----------------------------------------------------------------------------------------------- desenvolvimento

def comparacao_do_ensemble(media_m0: np.ndarray, media_mr: np.ndarray, y: np.ndarray, paineis: np.ndarray,
                           clusters: np.ndarray, *, limiares: dict[str, dict[str, Any]], replicas: int,
                           seed: int) -> dict[str, Any]:
    """O ensemble final no conjunto de selecao: metricas, delta MR - M0 com IC por cluster (os mesmos sorteios para
    os dois sistemas) e as metricas com o limiar do fold 1. DESCRITIVO: nao muda composicao nem limiar."""
    resumo = {"M0": metricas.resumo(media_m0, y, paineis), "MR": metricas.resumo(media_mr, y, paineis)}
    return {
        **resumo,
        "delta": {k: (None if resumo["M0"][k] is None or resumo["MR"][k] is None
                      else float(resumo["MR"][k] - resumo["M0"][k])) for k in ("macro", "auroc", "auprc")},
        "bootstrap": metricas.bootstrap_pareado_por_cluster(media_m0, media_mr, y, paineis, clusters,
                                                            replicas=replicas, seed=seed),
        "com_o_limiar_do_fold1": {s: {"threshold": limiares[s]["threshold"],
                                      **com_limiar(y, m, limiares[s]["threshold"])}
                                  for s, m in (("M0", media_m0), ("MR", media_mr))},
    }


COMO_LER_O_DESENVOLVIMENTO = (
    "EXPLORATORIO e descritivo: a composicao e a regra do limiar ja estavam declaradas, e o resultado nao muda nenhuma "
    "das duas (revisao de 24/09). Os ICs reamostram clusters do conjunto de selecao com os sistemas FIXOS: sao "
    "condicionais aos adapters e cabecas treinados e nao incluem a variacao de treino. O conjunto de selecao escolheu "
    "a extracao e a politica so com M0 e pode favorecer o M0, com vies de tamanho desconhecido. A macro do ensemble "
    "final nao e a media dos deltas dos pares; um IC que inclui zero nao demonstra equivalencia nem ausencia de "
    "degradacao. E classificacao geral, sem participacao brasileira: nao responde a pergunta regional (G7).")


# ----------------------------------------------------------------------------------------------- congelamento

def _declarado(estado: Any) -> bool:
    return str(estado).upper().startswith(("DECLARADO", "CONFIRMADO"))


def _finito(valor: Any) -> bool:
    return isinstance(valor, (int, float)) and not isinstance(valor, bool) and math.isfinite(valor)


def _texto(valor: Any) -> bool:
    return isinstance(valor, str) and bool(valor.strip())


ESTUDOS = ("br_clinical_evidence", "br_population_observed")
DELTAS = ("delta_br_full", "delta_br_matched", "delta_control")
METRICAS_DAS_MARGENS = ("auroc", "auprc")
#: Cada regra e uma lista de CONDICOES sobre um delta MR - M0, todas exigidas. Cada condicao compara a estimativa ou
#: o limite inferior do IC (p2_5) com um limite, por `>=` ou `>` (estrito: "o IC exclui zero" e `p2_5 > 0`). Assim
#: "estimativa >= 0,02 com IC excluindo zero" e "limite inferior >= 0,02" sao regras DIFERENTES e as duas se
#: escrevem (revisao de 25/09); a segunda exige evidencia mais forte.
ESTATISTICAS = ("estimativa", "p2_5")
COMPARACOES = (">=", ">")
#: As tres margens do Mosaic (PLAN 13.5) e a condicao 3 do protocolo, com o delta que cada uma pode usar e o sinal
#: do limite: melhoria pede limite >= 0; regressao maxima, limite <= 0.
MARGENS_EXIGIDAS: dict[str, tuple[tuple[str, ...], str]] = {
    "melhoria_minima_no_coorte_br": (("delta_br_full", "delta_br_matched"), ">= 0"),
    "regressao_maxima_no_controle": (("delta_control",), "<= 0"),
    "paineis_com_regressao_inaceitavel": (DELTAS, "<= 0"),
    "beneficio_nao_explicado_por_um_painel": (("delta_br_full", "delta_br_matched"), ">= 0"),
}
#: As condicoes 1 a 3 do Mosaic valem para todo estudo EXIGIDO; estudo DESCRITIVO so e relatado, sem regra.
CONDICOES_DO_MOSAIC = ("melhoria_minima_no_coorte_br", "regressao_maxima_no_controle",
                       "beneficio_nao_explicado_por_um_painel")
PAPEIS_DOS_ESTUDOS = ("exigido", "descritivo")
UNIDADES_IMPLEMENTADAS = ("cluster_conjunto", "par")


def _problemas_das_condicoes(nome: str, condicoes: Any, sinal: str) -> list[str]:
    if not (isinstance(condicoes, list) and condicoes):
        return [f"{nome}.condicoes: lista nao vazia de {{estatistica, comparacao, limite}}, recebeu {condicoes!r}"]
    problemas, vistas = [], set()
    for i, condicao in enumerate(condicoes):
        rotulo = f"{nome}.condicoes[{i}]"
        if not isinstance(condicao, dict):
            problemas.append(f"{rotulo}: objeto {{estatistica, comparacao, limite}}")
            continue
        for campo, dominio in (("estatistica", ESTATISTICAS), ("comparacao", COMPARACOES)):
            if condicao.get(campo) not in dominio:
                problemas.append(f"{rotulo}.{campo}: um de {list(dominio)}, recebeu {condicao.get(campo)!r}")
        limite = condicao.get("limite")
        if not _finito(limite):
            problemas.append(f"{rotulo}.limite: numero finito, recebeu {limite!r}")
        elif (sinal == ">= 0" and limite < 0) or (sinal == "<= 0" and limite > 0):
            problemas.append(f"{rotulo}.limite: tem de ser {sinal}, recebeu {limite}")
        if condicao.get("estatistica") in vistas:
            problemas.append(f"{nome}.condicoes: {condicao.get('estatistica')} repetida")
        vistas.add(condicao.get("estatistica"))
    return problemas


def _problemas_da_regra(nome: str, item: Any, deltas: tuple[str, ...] | None, sinal: str) -> list[str]:
    """Uma regra: estudos, delta, metrica e condicoes; `deltas` None = a regra nao escolhe delta (a interacao)."""
    if not isinstance(item, dict):
        return [f"{nome}: ausente"]
    problemas = []
    estudos = item.get("estudos")
    if not (isinstance(estudos, list) and estudos and len(set(estudos)) == len(estudos) and set(estudos) <= set(ESTUDOS)):
        problemas.append(f"{nome}.estudos: lista nao vazia de {list(ESTUDOS)}, recebeu {estudos!r}")
    campos = (("metrica", METRICAS_DAS_MARGENS),)
    for campo, dominio in ((("delta", deltas),) if deltas is not None else ()) + campos:
        if item.get(campo) not in dominio:
            problemas.append(f"{nome}.{campo}: um de {list(dominio)}, recebeu {item.get(campo)!r}")
    return problemas + _problemas_das_condicoes(nome, item.get("condicoes"), sinal)


def _problemas_dos_papeis(margens: dict[str, Any]) -> list[str]:
    """Cada estudo EXIGIDO tem as condicoes 1 a 3 do Mosaic; estudo DESCRITIVO nao aparece em regra nenhuma."""
    papeis = margens.get("papel_dos_estudos")
    if not (isinstance(papeis, dict) and set(papeis) == set(ESTUDOS) and set(papeis.values()) <= set(PAPEIS_DOS_ESTUDOS)):
        return [f"papel_dos_estudos: {{estudo: exigido ou descritivo}} para {list(ESTUDOS)}, recebeu {papeis!r}"]
    exigidos = [e for e in ESTUDOS if papeis[e] == "exigido"]
    descritivos = [e for e in ESTUDOS if papeis[e] == "descritivo"]
    problemas = [] if exigidos else ["papel_dos_estudos: ao menos um estudo exigido"]
    for nome in CONDICOES_DO_MOSAIC:
        estudos = (margens.get(nome) or {}).get("estudos") or []
        faltando = [e for e in exigidos if e not in estudos]
        if faltando:
            problemas.append(f"{nome}.estudos: as condicoes do Mosaic valem para todo estudo exigido; faltam {faltando}")
    for nome in (*MARGENS_EXIGIDAS, "interacao"):
        indevidos = [e for e in descritivos if e in ((margens.get(nome) or {}).get("estudos") or [])]
        if indevidos:
            problemas.append(f"{nome}.estudos: estudo descritivo nao tem regra ({indevidos})")
    return problemas


def problemas_das_margens(margens: dict[str, Any]) -> list[str]:
    """O que falta para as margens valerem como regra de decisao: estado declarado E, em cada uma, estudos, delta,
    metrica e condicoes com limite finito, mais o papel de cada estudo -- nao basta o texto do estado. Pura."""
    problemas = []
    for nome, (deltas, sinal) in MARGENS_EXIGIDAS.items():
        item = margens.get(nome)
        if nome == "paineis_com_regressao_inaceitavel" and isinstance(item, dict) and item.get("paineis") == []:
            if not _texto(item.get("motivo")):
                problemas.append(f"{nome}: lista vazia de paineis exige `motivo`")
            continue
        problemas += _problemas_da_regra(nome, item, deltas, sinal)
        if nome == "paineis_com_regressao_inaceitavel" and isinstance(item, dict):
            paineis = item.get("paineis")
            if not (isinstance(paineis, list) and paineis and set(paineis) <= set(metricas.PAINEIS_DE_DISCRIMINACAO)):
                problemas.append(f"{nome}.paineis: lista de {list(metricas.PAINEIS_DE_DISCRIMINACAO)} (ou vazia com "
                                 f"motivo), recebeu {paineis!r}")
        if nome == "beneficio_nao_explicado_por_um_painel" and isinstance(item, dict):
            suporte = item.get("suporte_minimo_por_painel")
            if not (isinstance(suporte, int) and not isinstance(suporte, bool) and suporte >= 1):
                problemas.append(f"{nome}.suporte_minimo_por_painel: inteiro >= 1 (P e B do painel), recebeu "
                                 f"{suporte!r}")
    interacao = margens.get("interacao")
    if not isinstance(interacao, dict) or not isinstance(interacao.get("criterio_proprio"), bool):
        problemas.append("interacao.criterio_proprio: true ou false, explicito")
    elif interacao["criterio_proprio"]:
        problemas += _problemas_da_regra("interacao", interacao, None, ">= 0")
    elif not _texto(interacao.get("motivo")):
        problemas.append("interacao: criterio_proprio false exige `motivo` (ex.: relatada com os absolutos)")
    problemas += _problemas_dos_papeis(margens)
    if not _declarado(margens.get("estado")):
        incompletas = sorted({p.split(".")[0].split(":")[0] for p in problemas})
        return [f"nao declaradas ({margens.get('estado')}); incompletas: {', '.join(incompletas) or 'nenhuma'}"]
    return problemas


def problemas_do_bootstrap(bootstrap: dict[str, Any]) -> list[str]:
    """Unidade principal e de sensibilidade entre as implementadas, replicas, seed e percentis. Pura."""
    if not _declarado(bootstrap.get("estado")):
        return [f"nao declarado ({bootstrap.get('estado')}); unidade_principal = {bootstrap.get('unidade_principal')!r}"]
    problemas = []
    principal, sensibilidade = bootstrap.get("unidade_principal"), bootstrap.get("unidade_de_sensibilidade")
    if principal not in UNIDADES_IMPLEMENTADAS:
        problemas.append(f"unidade_principal: uma de {list(UNIDADES_IMPLEMENTADAS)} (o componente conexo cluster-par "
                         f"nao esta implementado), recebeu {principal!r}")
    if sensibilidade not in UNIDADES_IMPLEMENTADAS or sensibilidade == principal:
        problemas.append(f"unidade_de_sensibilidade: a outra de {list(UNIDADES_IMPLEMENTADAS)}, recebeu "
                         f"{sensibilidade!r}")
    replicas, seed = bootstrap.get("replicas"), bootstrap.get("seed")
    if not (isinstance(replicas, int) and not isinstance(replicas, bool) and replicas >= 1000):
        problemas.append(f"replicas: inteiro >= 1000 (o Mosaic usa 1000), recebeu {replicas!r}")
    if not (isinstance(seed, int) and not isinstance(seed, bool)):
        problemas.append(f"seed: inteiro, recebeu {seed!r}")
    if bootstrap.get("percentis") != [2.5, 97.5]:
        problemas.append(f"percentis: [2.5, 97.5], recebeu {bootstrap.get('percentis')!r}")
    return problemas


def problemas_das_pendencias(pendencias: Iterable[dict[str, Any]]) -> list[str]:
    """Pendencia resolvida e FEITO com `onde` (arquivo, commit ou teste) ou RETIRADO com `motivo`. Pura."""
    problemas = []
    for pendencia in pendencias:
        estado, item = pendencia.get("estado"), pendencia.get("item")
        if estado == "FEITO" and not _texto(pendencia.get("onde")):
            problemas.append(f"pendencia FEITO sem `onde`: {item}")
        elif estado == "RETIRADO" and not _texto(pendencia.get("motivo")):
            problemas.append(f"pendencia RETIRADO sem `motivo`: {item}")
        elif estado not in ESTADOS_RESOLVIDOS:
            problemas.append(f"pendencia {estado}: {item}")
    return problemas


def problemas_do_codigo(estado: dict[str, Any]) -> list[str]:
    """Revisao do git conferida (40 hex), nenhum arquivo de codigo ausente, fora do git ou com mudanca. Um git que
    falha NAO vale como `sem mudancas` (revisao de 24/09). Pura."""
    if estado.get("erro"):
        return [f"estado do codigo nao conferido (git falhou: {estado['erro']})"]
    problemas = []
    revisao = str(estado.get("revisao") or "")
    if len(revisao) != 40 or any(c not in "0123456789abcdef" for c in revisao):
        problemas.append(f"revisao do git nao conferida: {revisao!r}")
    problemas += [f"codigo ausente: {a}" for a in estado.get("ausentes", [])]
    problemas += [f"codigo fora do git: {a}" for a in estado.get("nao_rastreados", [])]
    problemas += [f"codigo com mudanca nao commitada: {a}" for a in estado.get("modificados", [])]
    return problemas


def bloqueios(campanha: dict[str, Any], *, estado_do_codigo: dict[str, Any],
              proveniencia: dict[str, dict[str, Any]] | None = None,
              entradas: dict[str, dict[str, Any]] | None = None) -> list[str]:
    """O que impede o congelamento, na ordem em que se resolve. Vazio = pode congelar. Pura."""
    g6 = campanha["g6"]
    saida = [f"margens: {p}" for p in problemas_das_margens(g6["margens"])]
    saida += [f"bootstrap da interacao: {p}" for p in problemas_do_bootstrap(g6["bootstrap_da_interacao"])]
    saida += problemas_das_pendencias(g6.get("pendencias_antes_do_congelamento", []))
    saida += problemas_do_codigo(estado_do_codigo)
    if not ((proveniencia or {}).get("abraom") or {}).get("confere"):
        saida.append("abraom_snapshot_hash nao reconferido no arquivo (--proveniencia abraom=<SABE1171.Abraom.clean.tsv>)")
    for nome, descricao in ENTRADAS_DAS_ANALISES_SECUNDARIAS.items():
        if not ((entradas or {}).get(nome) or {}).get("sha256"):
            saida.append(f"entrada da analise secundaria sem registro (--entrada {nome}=<arquivo>): {descricao}")
    return saida


def metodo_de_regionalizacao(campanha: dict[str, Any], *, modulos: int) -> str:
    """O `regionalization_method` do manifesto, escrito a partir da receita declarada (fonte unica)."""
    receita = campanha["adapter_do_mr"]["receita"]
    lora = receita["lora"]
    plano = campanha["g6"]["proveniencia"]["dados_do_adapter"]["plano"]
    return (f"{'rsLoRA' if lora['rslora'] else 'LoRA'} rank {lora['rank']}, alpha {lora['alpha']}, dropout "
            f"{lora['dropout']}, em {modulos} modulos ({receita['modulos_esperados']}) do R03 congelado; MLM em "
            f"janelas de {receita['window_bp']} bp com alelos do plano de janelas ({plano['mistura']}; ABraOM "
            f"SABE1171 e gnomAD v4.1 joint), separado por loco em treino e validacao; {receita['passos']} passos de "
            f"{receita['exemplos_por_passo']} exemplos, lr {receita['lr']}, pesos {receita['pesos']}; checkpoint "
            f"pelo menor focal_alt na validacao do adapter; sementes "
            f"{', '.join(map(str, campanha['sementes']['adapter']))}")


def montar_manifesto(campanha: dict[str, Any], *, declaracao_sha256: str, decisao_g5: dict[str, Any],
                     componentes: list[dict[str, Any]], caches: dict[str, dict[str, Any]],
                     extracao: dict[str, Any], limiares: dict[str, dict[str, Any]], fold1: dict[str, Any],
                     conferencias: dict[str, dict[str, Any]], proveniencia: dict[str, dict[str, Any]],
                     codigo: dict[str, Any], ambiente_da_pontuacao: dict[str, Any], modulos: int,
                     bloqueios_atuais: list[str], criado_em_utc: str, estado: str = RASCUNHO,
                     entradas: dict[str, dict[str, Any]] | None = None,
                     raiz_dos_comparadores: str | None = None) -> dict[str, Any]:
    """O manifesto do G6: tudo o que o G7 precisa para pontuar o sistema congelado e tudo o que foi aplicado, com o
    que foi reconferido separado do que so esta declarado. O rascunho e o congelado so diferem no `estado`. Pura."""
    if estado not in (RASCUNHO, CONGELADO):
        raise ValueError(f"estado {estado!r}")
    if estado == CONGELADO and bloqueios_atuais:
        raise ManifestoInvalido(f"com bloqueios o manifesto nao congela: {bloqueios_atuais}")
    g6 = campanha["g6"]
    pre = g6["proveniencia"]["pre_treino_do_r03"]
    da_cabeca = g6["proveniencia"]["treino_da_cabeca"]
    abraom = g6["proveniencia"]["abraom"]
    checkpoints = sorted({c["checkpoint_sha256"] for c in caches.values()})
    por_sistema = {s: [c for c in componentes if c["sistema"] == s] for s in SISTEMAS}
    cabecas = {s: [{"arquivo": c["arquivo"], "sha256": c["sha256"]} for c in por_sistema[s]] for s in SISTEMAS}
    backbone = {"checkpoint_id": pre["checkpoint_id"], "arquivo": pre["arquivo"], "passo": pre["passo"],
                "sha256": checkpoints[0] if len(checkpoints) == 1 else checkpoints}
    adapters = [{"semente": c["adapter"], "sha256": caches[str(c["adapter"])]["adapter_sha256"]}
                for c in por_sistema["MR"]]
    return {
        "formato": FORMATO_DO_MANIFESTO,
        "estado": estado,
        "bloqueios": list(bloqueios_atuais),
        "criado_em_utc": criado_em_utc,
        "natureza": g6["natureza"],
        "declaracao": {"arquivo": "configs/campanha_r03_desenvolvimento.json", "sha256": declaracao_sha256},
        "decisao_do_g5": decisao_g5,
        "campos_do_mosaic": {
            "base_checkpoint_id": {"backbone": backbone, "cabecas": cabecas["M0"],
                                   "leitura": "sistema base = R03 congelado + cabeca clinica; tres componentes"},
            "regionalized_checkpoint_id": {"backbone": backbone, "adapters": adapters, "cabecas": cabecas["MR"],
                                           "leitura": "o mesmo R03 + adapter populacional + cabeca clinica"},
            "base_training_dataset_id": {"pre_treino_do_r03": pre["dataset_id"],
                                         "treino_da_cabeca": da_cabeca["dataset_id"].replace(
                                             "<politica>", decisao_g5["politica"])},
            "base_training_dataset_hash": {"pre_treino_do_r03": pre["dataset_hash"],
                                           "treino_da_cabeca": decisao_g5["snapshot_sha256"]},
            "base_training_cutoff": {"pre_treino_do_r03": pre["cutoff"],
                                     "rotulos_da_cabeca": da_cabeca["cutoff_dos_rotulos"]},
            "abraom_snapshot_hash": {"sha256": abraom["sha256"], "fonte": abraom["fonte"],
                                     "reconferido_no_arquivo": bool((proveniencia.get("abraom") or {})
                                                                    .get("confere"))},
            "regionalization_method": metodo_de_regionalizacao(campanha, modulos=modulos),
            "regionalized_training_dataset": "same_as_base: as cabecas do MR treinam no mesmo snapshot (sha256 "
                                             "conferido nas seis cabecas) que as do M0",
            "regionalized_base_checkpoint": "same_as_base: o mesmo R03 (checkpoint_sha256 igual nos quatro caches)",
            "study_membership_used_for_training": False,
            "study_labels_used_for_training": False,
            "nota_da_membership": "a membership dos estudos so foi usada para EXCLUIR (membros, alelos, vizinhanca "
                                  "de 2.048 bp), nunca como exemplo de treino",
            "origem_do_treino_da_cabeca": da_cabeca["origem"],
            "nota_do_pre_treino": pre["nota"],
        },
        "sistemas": {
            NO_MOSAIC[s]: {
                "nome": s,
                "componentes": [{k: c[k] for k in ("rotulo", "cabeca", "adapter", "arquivo", "sha256", "platt",
                                                   "epoca", "cache")} for c in por_sistema[s]],
                "predicao": g6["predicao_do_sistema"],
                "limiar": limiares[s],
            } for s in SISTEMAS},
        "limiar_do_ensemble": {"regra": g6["regra_do_limiar"], "conjunto": fold1,
                               "disjuncao_com_os_estudos": "por construcao: tabela_de_extracao recusa membros dos "
                                                           "estudos nos caches de desenvolvimento; o G7 reconfere"},
        "caches_de_desenvolvimento": caches,
        "extracao_dos_caches": extracao,
        "conferencias_das_cabecas": conferencias,
        "adapters_congelados": {str(a["semente"]): campanha["adapters_congelados"][str(a["semente"])]
                                for a in adapters},
        "exclusoes_aplicadas": g6["exclusoes_aplicadas"],
        "sobreposicoes_declaradas": g6["sobreposicoes_declaradas"],
        "proveniencia": {"declarada": g6["proveniencia"], "reconferida_nos_arquivos": proveniencia},
        "margens": g6["margens"],
        "bootstrap": {"replicas": int(g6["bootstrap_da_interacao"].get("replicas") or 1000),
                      "seed": int(g6["bootstrap_da_interacao"].get("seed") or 20260901),
                      "unidade": "overlap_cluster_id", "percentis": [2.5, 97.5],
                      "unidade_principal_da_interacao": g6["bootstrap_da_interacao"].get("unidade_principal"),
                      "interacao": g6["bootstrap_da_interacao"],
                      "uso": "o G7 real usa estas replicas, esta seed e esta unidade principal; a linha de comando "
                             "nao as muda",
                      "leitura": "ICs condicionais aos sistemas congelados: reamostram variantes (por cluster), nao o "
                                 "treino dos adapters e das cabecas"},
        "analises_secundarias": g6["analises_secundarias"],
        "entradas_das_analises_secundarias": entradas or {},
        "raiz_dos_comparadores": raiz_dos_comparadores,
        "pendencias": g6.get("pendencias_antes_do_congelamento", []),
        "codigo": codigo,
        "ambiente_da_pontuacao": ambiente_da_pontuacao,
    }


def serializar(manifesto: dict[str, Any]) -> bytes:
    """Bytes canonicos: chaves ordenadas, UTF-8, sem NaN. O sha256 do manifesto e o destes bytes."""
    return (json.dumps(manifesto, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def gravar_manifesto(pasta: Path, manifesto: dict[str, Any]) -> str:
    """Grava o manifesto CONGELADO e, a parte, o sha256 dos bytes (formato do sha256sum). Nunca sobrescreve."""
    if manifesto.get("estado") != CONGELADO or manifesto.get("bloqueios"):
        raise ManifestoInvalido(f"so um manifesto sem bloqueios congela: {manifesto.get('bloqueios')}")
    pasta = Path(pasta)
    caminho, do_sha = pasta / NOME_DO_MANIFESTO, pasta / f"{NOME_DO_MANIFESTO}.sha256"
    if caminho.exists() or do_sha.exists():
        raise ManifestoInvalido(f"{caminho} ja existe: um manifesto congelado nao se sobrescreve")
    dados = serializar(manifesto)
    sha = hashlib.sha256(dados).hexdigest()
    caminho.write_bytes(dados)
    do_sha.write_text(f"{sha}  {NOME_DO_MANIFESTO}\n", encoding="utf-8")
    return sha


def ler_manifesto_congelado(pasta: Path) -> tuple[dict[str, Any], str]:
    """(manifesto, sha256) de um G6 congelado. Recusa bytes que nao conferem com o sha256 a parte, outro formato ou
    manifesto com bloqueios -- e o que o G7 usa antes de extrair ou pontuar qualquer coisa."""
    pasta = Path(pasta).expanduser()
    caminho, do_sha = pasta / NOME_DO_MANIFESTO, pasta / f"{NOME_DO_MANIFESTO}.sha256"
    if not caminho.exists() or not do_sha.exists():
        raise ManifestoInvalido(f"{pasta}: sem {NOME_DO_MANIFESTO} ou sem o sha256 a parte")
    dados = caminho.read_bytes()
    sha = hashlib.sha256(dados).hexdigest()
    declarado = (do_sha.read_text(encoding="utf-8").split() or [""])[0]
    if sha != declarado:
        raise ManifestoInvalido(f"{caminho}: sha256 {sha[:12]} != o declarado {declarado[:12]}")
    manifesto = json.loads(dados.decode("utf-8"))
    if manifesto.get("formato") != FORMATO_DO_MANIFESTO:
        raise ManifestoInvalido(f"formato {manifesto.get('formato')!r}, esperado {FORMATO_DO_MANIFESTO!r}")
    if manifesto.get("estado") != CONGELADO or manifesto.get("bloqueios"):
        raise ManifestoInvalido(f"manifesto nao congelado: {manifesto.get('estado')}, {manifesto.get('bloqueios')}")
    return manifesto, sha
