"""G7: o consumidor dos estudos brasileiros -- coortes, cobertura, deltas, interacao e bootstrap pareado. Sem torch.

Recebe as probabilidades de dois sistemas JA congelados (G6) e mede. Nada aqui escolhe, ajusta ou calibra.

Aplica as regras de AVALIACAO do Mosaic. A campanha, como um todo, e um protocolo DERIVADO -- a cabeca e treinada e
calibrada no `core_locus` do release, com exclusoes proprias (autorizado pelo mantenedor em 15/09) -- e deve ser
descrita assim, nao como cumprimento integral do protocolo publicado.

REGRAS DE AVALIACAO DO MOSAIC (commit 814e7f0: `specs/PLAN.md` 13.3-13.5 e `protocol.py:brazil_protocol_section`)
    - cada estudo separado: `br_clinical_evidence` e `br_population_observed` nunca se unem;
    - coorte completo = `case` + `unmatched_case` (delta_br_full); casos pareados = `case` com controle
      bidirecional (delta_br_matched); controles = `control` (delta_control); interacao = delta_br_matched -
      delta_control, SEM `unmatched_case`;
    - AUROC e AUPRC (precisao media, a `average_precision_score` do Mosaic) so com as duas classes; n_P, n_B e
      cobertura sempre; deltas na INTERSECAO de cobertura dos dois sistemas; variante sem score entra na cobertura
      e nunca e imputada;
    - o relatorio do coorte inteiro e obrigatorio; os paineis de discriminacao sao diagnostico; nao ha macro
      brasileira; plof e synonymous sao guarda; sem piso 50/50;
    - metricas com limiar so com limiar externo congelado por sistema; sem ele, sao omitidas;
    - bootstrap pareado por `overlap_cluster_id` (os MESMOS sorteios para os dois sistemas), 1.000 replicas, seed
      20260901, percentis 2,5 e 97,5.

O QUE O MOSAIC NAO DEFINE E FICA DECLARADO AQUI (plano, secao 6.3; PROPOSTO ate o G6)
    - a reamostragem da interacao, com dois metodos de pressupostos diferentes, relatados juntos:
        * clusters sorteados EM CONJUNTO sobre a uniao de casos pareados e controles: preserva a dependencia
          genomica (um cluster entra inteiro), mas NAO preserva os pares -- caso e controle em clusters diferentes
          sao sorteados independentemente;
        * sorteio por PAR: preserva o pareamento, mas nao a dependencia entre pares do mesmo cluster;
    - cada analise tem o proprio gerador, com a seed declarada, para o resultado de uma nao depender de quais outras
      rodam;
    - a interacao SUBTRAI os deltas observados; nao remove confundimento nem diferencas de composicao entre casos e
      controles;
    - no estudo clinico, o subconjunto `present_abraom` do coorte completo (exigido pelo Mosaic);
    - sensibilidades pre-declaradas (plano 6.4 e achado de 20/09), sempre sem desfazer pares -- o controle sai junto
      com o seu caso: pares com caso E controle fora do ABraOM; sem os pares cujo controle tem SCV brasileira (regra
      ampla); pares com exposicao de locus empatada; e, se a cobertura desfizer pares, so os pares completos. Entrada
      ausente nao faz a analise sumir: ela sai marcada como nao calculada, com o motivo;
    - Brier, so dos SISTEMAS e so sobre probabilidades calibradas (`com_brier`); nunca sobre um score de ordenacao;
    - baselines (`avaliar_baseline`): scores FORA dos sistemas, com metricas absolutas na propria cobertura; score
      constante num coorte nao discrimina por construcao e sai sem metrica, com o motivo.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from eval.campanha import metricas

ESTUDO_CLINICO = "br_clinical_evidence"
ESTUDO_POPULACIONAL = "br_population_observed"
ESTUDOS = (ESTUDO_CLINICO, ESTUDO_POPULACIONAL)
CASO, SEM_PAR, CONTROLE = "case", "unmatched_case", "control"
COORTE_COMPLETO, CASOS_PAREADOS, CONTROLES = "full_cohort", "matched_cases", "controls"
BASE, REGIONALIZADO = "base", "regionalized"
SISTEMAS = (BASE, REGIONALIZADO)
CONTINUAS = ("auroc", "auprc")
PAINEIS_DE_DISCRIMINACAO = metricas.PAINEIS_DE_DISCRIMINACAO
PAINEIS_DE_GUARDA = metricas.PAINEIS_DE_GUARDA
REPLICAS = 1000
SEED = 20260901
PERCENTIS = (2.5, 97.5)
COLUNAS_DOS_MEMBROS = ("variant_id", "study_id", "member_role", "matched_variant_id", "binary_label",
                       "primary_panel", "overlap_cluster_id", "present_abraom")


class EstudoInvalido(ValueError):
    """O membership nao e o que o protocolo promete: nada e medido."""


# ------------------------------------------------------------------------------------------------ coortes

def _parceiro(valor: Any) -> str | None:
    return None if valor is None or pd.isna(valor) or str(valor) == "" else str(valor)


def visoes(membros: pd.DataFrame, estudo: str) -> dict[str, pd.DataFrame]:
    """As tres visoes do Mosaic (`comparator_eval/cohorts.py:brazil_views`), com o pareamento CONFERIDO.

    O G1 ja validou o membership; aqui a conferencia se repete porque a interacao depende dela: caso sem controle
    bidirecional, controle orfao ou `unmatched_case` com par mudariam a interacao em silencio.
    """
    faltando = [c for c in COLUNAS_DOS_MEMBROS if c not in membros.columns]
    if faltando:
        raise EstudoInvalido(f"colunas ausentes no membership: {faltando}")
    if estudo not in ESTUDOS:
        raise EstudoInvalido(f"estudo {estudo!r} fora de {ESTUDOS}")
    linhas = membros[membros["study_id"] == estudo].reset_index(drop=True)
    if linhas.empty:
        raise EstudoInvalido(f"{estudo} sem membros")
    ids = linhas["variant_id"].astype(str)
    if ids.duplicated().any():
        raise EstudoInvalido(f"{estudo}: variante com mais de um papel, ex.: {ids[ids.duplicated()].tolist()[:3]}")
    papeis = set(linhas["member_role"].astype(str))
    if papeis - {CASO, SEM_PAR, CONTROLE}:
        raise EstudoInvalido(f"{estudo}: papeis fora do protocolo {sorted(papeis - {CASO, SEM_PAR, CONTROLE})}")
    clusters = linhas["overlap_cluster_id"]
    if clusters.isna().any() or (clusters.astype(str).str.strip() == "").any():
        raise EstudoInvalido(f"{estudo}: membro sem overlap_cluster_id (a unidade do bootstrap)")
    if not set(linhas["binary_label"].astype(int)) <= {0, 1}:
        raise EstudoInvalido(f"{estudo}: binary_label fora de {{0, 1}}")
    if linhas["present_abraom"].isna().any():
        raise EstudoInvalido(f"{estudo}: present_abraom nulo (o subconjunto e a sensibilidade dependem dele)")

    papel_de = dict(zip(ids, linhas["member_role"].astype(str)))
    par_de = {vid: _parceiro(par) for vid, par in zip(ids, linhas["matched_variant_id"])}
    problemas = []
    for vid, papel in papel_de.items():
        par = par_de[vid]
        if papel == SEM_PAR:
            if par is not None:
                problemas.append(f"unmatched_case {vid} com par {par}")
            continue
        esperado = CONTROLE if papel == CASO else CASO
        if par is None or papel_de.get(par) != esperado or par_de.get(par) != vid:
            problemas.append(f"{papel} {vid} sem par bidirecional ({par})")
    if problemas:
        raise EstudoInvalido(f"{estudo}: {len(problemas)} problema(s) de pareamento, ex.: {problemas[:3]}")
    papel = linhas["member_role"].astype(str)
    return {COORTE_COMPLETO: linhas[papel.isin([CASO, SEM_PAR])].reset_index(drop=True),
            CASOS_PAREADOS: linhas[papel == CASO].reset_index(drop=True),
            CONTROLES: linhas[papel == CONTROLE].reset_index(drop=True)}


def composicao(frame: pd.DataFrame) -> dict[str, Any]:
    """n, P e B no coorte e por painel, ANTES da cobertura."""
    y = frame["binary_label"].astype(int)
    por_painel = {}
    for painel, grupo in frame.groupby("primary_panel", sort=True):
        rotulos = grupo["binary_label"].astype(int)
        por_painel[str(painel)] = {"n": int(len(grupo)), "n_P": int(rotulos.sum()),
                                   "n_B": int((rotulos == 0).sum())}
    return {"n": int(len(frame)), "n_P": int(y.sum()), "n_B": int((y == 0).sum()),
            "clusters": int(frame["overlap_cluster_id"].nunique()), "por_painel": por_painel}


# ------------------------------------------------------------------------------------------------ metricas

def suporte(y: np.ndarray, pontuadas: np.ndarray) -> dict[str, Any]:
    """Contagens e cobertura como o Mosaic (`stats.support_counts`)."""
    y = np.asarray(y, dtype=int)
    n_total, n_p, n_b = int(y.size), int((y == 1).sum()), int((y == 0).sum())
    n_s = int(pontuadas.sum())
    n_ps, n_bs = int(((y == 1) & pontuadas).sum()), int(((y == 0) & pontuadas).sum())
    return {"n_total": n_total, "n_P_total": n_p, "n_B_total": n_b, "n_scored": n_s, "n_P_scored": n_ps,
            "n_B_scored": n_bs, "coverage": n_s / n_total if n_total else None,
            "coverage_P": n_ps / n_p if n_p else None, "coverage_B": n_bs / n_b if n_b else None,
            "prevalence_scored": n_ps / n_s if n_s else None}


def continuas(y: np.ndarray, s: np.ndarray) -> dict[str, float | None]:
    """AUROC e AUPRC (precisao media); None sem as duas classes."""
    return {"auroc": metricas.auroc(s, y), "auprc": metricas.auprc(s, y)}


def brier(y: np.ndarray, p: np.ndarray) -> float | None:
    """Media de (p - y)^2 sobre PROBABILIDADES calibradas; None sem linha. Menor e melhor."""
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    return float(np.mean((p - y) ** 2)) if y.size else None


def com_limiar(y: np.ndarray, s: np.ndarray, limiar: float) -> dict[str, float | None]:
    """MCC, sensibilidade e especificidade com a regra do Mosaic (`score >= limiar`; MCC 0 com denominador 0)."""
    y = np.asarray(y, dtype=int)
    previsto = np.asarray(s, dtype=float) >= limiar
    vp, vn = int(((y == 1) & previsto).sum()), int(((y == 0) & ~previsto).sum())
    fp, fn = int(((y == 0) & previsto).sum()), int(((y == 1) & ~previsto).sum())
    denominador = (vp + fp) * (vp + fn) * (vn + fp) * (vn + fn)
    if not y.size:
        mcc = None
    else:
        mcc = (vp * vn - fp * fn) / float(denominador) ** 0.5 if denominador else 0.0
    return {"mcc": mcc, "sensitivity": vp / (vp + fn) if vp + fn else None,
            "specificity": vn / (vn + fp) if vn + fp else None,
            "n_false_negative": float(fn), "n_false_positive": float(fp)}


def _delta(a: float | None, b: float | None) -> float | None:
    return None if a is None or b is None else float(b - a)


def _chaves(limiares: dict[str, float] | None, com_brier: bool) -> tuple[str, ...]:
    return (CONTINUAS + (("brier",) if com_brier else ())
            + (("mcc", "sensitivity", "specificity") if limiares is not None else ()))


def comparar(y: np.ndarray, base: np.ndarray, regionalizado: np.ndarray,
             limiares: dict[str, float] | None = None, *, com_brier: bool = False) -> dict[str, Any]:
    """Cada sistema na INTERSECAO de cobertura e o delta regionalizado - base; a cobertura de cada sistema no
    proprio coorte vem a parte, porque variante sem score conta como nao coberta. No Brier, delta NEGATIVO e
    melhora."""
    y = np.asarray(y, dtype=int)
    base, regionalizado = np.asarray(base, dtype=float), np.asarray(regionalizado, dtype=float)
    ambos = np.isfinite(base) & np.isfinite(regionalizado)
    saida: dict[str, Any] = {"intersecao": suporte(y, ambos),
                             "cobertura": {BASE: suporte(y, np.isfinite(base)),
                                           REGIONALIZADO: suporte(y, np.isfinite(regionalizado))}}
    for sistema, s in ((BASE, base), (REGIONALIZADO, regionalizado)):
        saida[sistema] = continuas(y[ambos], s[ambos])
        if com_brier:
            saida[sistema]["brier"] = brier(y[ambos], s[ambos])
        if limiares is not None:
            saida[sistema].update(com_limiar(y[ambos], s[ambos], limiares[sistema]))
    saida["delta"] = {k: _delta(saida[BASE][k], saida[REGIONALIZADO][k]) for k in _chaves(limiares, com_brier)}
    return saida


# ------------------------------------------------------------------------------------------------ bootstrap

def grupos(clusters: np.ndarray) -> list[np.ndarray]:
    """Linhas de cada cluster (o `unique_unit_groups` do Mosaic, sem descartar linha: cluster vazio foi recusado
    nas visoes)."""
    _, inverso = np.unique(np.asarray(clusters).astype(str), return_inverse=True)
    ordem = np.argsort(inverso, kind="stable")
    limites = np.cumsum(np.bincount(inverso))[:-1]
    return np.split(ordem, limites)


def sortear(grupos_: list[np.ndarray], rng: np.random.Generator) -> np.ndarray:
    """Clusters com reposicao, todas as linhas de cada um (o `resample_indices` do Mosaic)."""
    escolhidos = rng.integers(0, len(grupos_), size=len(grupos_))
    return np.concatenate([grupos_[i] for i in escolhidos])


def intervalo(valores: list[float]) -> dict[str, Any]:
    if not valores:
        return {"p2_5": None, "p97_5": None, "replicas_validas": 0}
    baixo, alto = np.percentile(np.asarray(valores, dtype=float), PERCENTIS)
    return {"p2_5": float(baixo), "p97_5": float(alto), "replicas_validas": len(valores)}


class _Acumulador:
    def __init__(self) -> None:
        self.valores: dict[tuple, list[float]] = {}

    def guardar(self, chave: tuple, valor: float | None) -> None:
        lista = self.valores.setdefault(chave, [])
        if valor is not None and np.isfinite(valor):
            lista.append(float(valor))

    def intervalo(self, chave: tuple) -> dict[str, Any]:
        return intervalo(self.valores.get(chave, []))


# ------------------------------------------------------------------------------------------------ analises

def _arrays(frame: pd.DataFrame, pontos: dict[str, pd.Series]) -> dict[str, np.ndarray]:
    ids = frame["variant_id"].astype(str)
    return {"y": frame["binary_label"].astype(int).to_numpy(),
            BASE: pontos[BASE].reindex(ids).to_numpy(dtype=float),
            REGIONALIZADO: pontos[REGIONALIZADO].reindex(ids).to_numpy(dtype=float),
            "clusters": frame["overlap_cluster_id"].astype(str).to_numpy(),
            "paineis": frame["primary_panel"].astype(str).to_numpy()}


def analise_do_coorte(frame: pd.DataFrame, pontos: dict[str, pd.Series], *, replicas: int = REPLICAS,
                      seed: int = SEED, limiares: dict[str, float] | None = None,
                      por_painel: bool = False, com_brier: bool = False) -> dict[str, Any]:
    """Um coorte: sistemas e delta na intersecao, com IC do bootstrap pareado por cluster (os MESMOS sorteios para
    os dois sistemas). Com `por_painel`, repete por painel (diagnostico) e mede o delta SEM cada painel com suporte
    -- a informacao da condicao "beneficio nao explicado por um unico painel"."""
    a = _arrays(frame, pontos)
    celulas: dict[str, np.ndarray] = {"coorte": np.ones(len(frame), dtype=bool)}
    if por_painel:
        for painel in sorted(set(a["paineis"])):
            celulas[f"painel:{painel}"] = a["paineis"] == painel
        for painel in PAINEIS_DE_DISCRIMINACAO + PAINEIS_DE_GUARDA:
            if (a["paineis"] == painel).any() and (a["paineis"] != painel).any():
                celulas[f"sem_painel:{painel}"] = a["paineis"] != painel

    def medir(indices: np.ndarray) -> dict[str, dict[str, Any]]:
        saida = {}
        for nome, mascara in celulas.items():
            linhas = indices[mascara[indices]]
            resultado = comparar(a["y"][linhas], a[BASE][linhas], a[REGIONALIZADO][linhas], limiares,
                                 com_brier=com_brier)
            painel = nome[len("painel:"):] if nome.startswith("painel:") else None
            if painel is not None and painel not in PAINEIS_DE_DISCRIMINACAO:
                # Como no Mosaic: AUROC e AUPRC so nos paineis de discriminacao; plof e synonymous sao guarda
                # (contagem e metricas com limiar), `other` e descritivo.
                for sistema in SISTEMAS + ("delta",):
                    resultado[sistema]["auroc"] = resultado[sistema]["auprc"] = None
            saida[nome] = resultado
        return saida

    observado = medir(np.arange(len(frame)))
    acumulador = _Acumulador()
    todas = grupos(a["clusters"])
    rng = np.random.default_rng(seed)
    chaves = _chaves(limiares, com_brier)
    for _ in range(replicas):
        replica = medir(sortear(todas, rng))
        for nome, resultado in replica.items():
            for chave in chaves:
                for sistema in SISTEMAS:
                    acumulador.guardar((nome, sistema, chave), resultado[sistema].get(chave))
                acumulador.guardar((nome, "delta", chave), resultado["delta"].get(chave))

    saida: dict[str, Any] = {"composicao": composicao(frame)}
    for nome, resultado in observado.items():
        for sistema in SISTEMAS + ("delta",):
            for chave in chaves:
                valor = resultado[sistema].get(chave)
                resultado[sistema][chave] = {"estimativa": valor, **acumulador.intervalo((nome, sistema, chave))}
        saida[nome] = resultado
    return saida


def _indices_do_par(casos: pd.DataFrame, controles: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Para cada caso, a linha do seu controle (o pareamento ja foi conferido nas visoes)."""
    linha_do_controle = {vid: i for i, vid in enumerate(controles["variant_id"].astype(str))}
    parceiros = [linha_do_controle[str(par)] for par in casos["matched_variant_id"]]
    return np.arange(len(casos)), np.asarray(parceiros, dtype=int)


def interacao(casos: pd.DataFrame, controles: pd.DataFrame, pontos: dict[str, pd.Series], *,
              replicas: int = REPLICAS, seed: int = SEED) -> dict[str, Any]:
    """delta_br_matched - delta_control, cada delta na intersecao de cobertura do proprio grupo (regra do Mosaic).

    Dois ICs, com pressupostos diferentes (PROPOSTO, plano 6.3), sempre com os mesmos sorteios para os dois
    sistemas: (1) clusters sorteados EM CONJUNTO sobre a uniao de casos pareados e controles -- preserva a
    dependencia genomica, nao os pares (caso e controle em clusters diferentes saem em sorteios independentes);
    (2) sorteio por PAR -- preserva o pareamento, nao a dependencia entre pares do mesmo cluster.

    Com cobertura incompleta, a intersecao de cada grupo pode deixar casos e controles que ja nao correspondem par a
    par: por isso sai a contagem de pares com os dois membros cobertos pelos dois sistemas.
    """
    if len(casos) != len(controles):
        raise EstudoInvalido(f"{len(casos)} casos pareados e {len(controles)} controles: o pareamento e 1:1")
    ac, ak = _arrays(casos, pontos), _arrays(controles, pontos)
    linhas_caso, linhas_controle = _indices_do_par(casos, controles)
    coberto_caso = np.isfinite(ac[BASE]) & np.isfinite(ac[REGIONALIZADO])
    coberto_controle = np.isfinite(ak[BASE]) & np.isfinite(ak[REGIONALIZADO])
    pares_cobertos = int((coberto_caso[linhas_caso] & coberto_controle[linhas_controle]).sum())

    def estimar(ic: np.ndarray, ik: np.ndarray) -> dict[str, dict[str, float | None]]:
        dm = comparar(ac["y"][ic], ac[BASE][ic], ac[REGIONALIZADO][ic])["delta"]
        dc = comparar(ak["y"][ik], ak[BASE][ik], ak[REGIONALIZADO][ik])["delta"]
        return {k: {"delta_br_matched": dm[k], "delta_control": dc[k], "interacao": _delta(dc[k], dm[k])}
                for k in CONTINUAS}

    observado = estimar(np.arange(len(casos)), np.arange(len(controles)))

    # Uniao: as primeiras len(casos) linhas sao casos, as demais controles.
    clusters = np.concatenate([ac["clusters"], ak["clusters"]])
    conjunta, por_par = _Acumulador(), _Acumulador()
    todas = grupos(clusters)
    rng = np.random.default_rng(seed)
    for _ in range(replicas):
        indices = sortear(todas, rng)
        ic, ik = indices[indices < len(casos)], indices[indices >= len(casos)] - len(casos)
        replica = estimar(ic, ik) if ic.size and ik.size else {k: {} for k in CONTINUAS}
        for k in CONTINUAS:
            for nome in ("delta_br_matched", "delta_control", "interacao"):
                conjunta.guardar((k, nome), replica[k].get(nome))

    rng_par = np.random.default_rng(seed)
    for _ in range(replicas):
        pares = rng_par.integers(0, len(linhas_caso), size=len(linhas_caso))
        replica = estimar(linhas_caso[pares], linhas_controle[pares])
        for k in CONTINUAS:
            por_par.guardar((k, "interacao"), replica[k].get("interacao"))

    saida: dict[str, Any] = {
        "definicao": "delta_br_matched - delta_control (sem unmatched_case), deltas regionalized - base; subtrai "
                     "os deltas observados, nao remove confundimento nem diferencas de composicao",
        "reamostragem": {"principal": "overlap_cluster_id em CONJUNTO sobre casos pareados + controles: preserva a "
                                      "dependencia genomica, nao os pares",
                         "sensibilidade": "por par: preserva o pareamento, nao a dependencia entre pares do mesmo "
                                          "cluster",
                         "estado": "PROPOSTO (plano 6.3); mesmos sorteios para os dois sistemas nos dois metodos"},
        "pares": int(len(casos)),
        "pares_com_os_dois_membros_cobertos": pares_cobertos,
        "cobertura_desfaz_pares": pares_cobertos < len(casos),
        "clusters_na_uniao": int(len(todas)),
    }
    for k in CONTINUAS:
        saida[k] = {nome: {"estimativa": observado[k][nome], **conjunta.intervalo((k, nome))}
                    for nome in ("delta_br_matched", "delta_control", "interacao")}
        saida[k]["interacao_sensibilidade_por_par"] = {"estimativa": observado[k]["interacao"],
                                                        **por_par.intervalo((k, "interacao"))}
    return saida


def controles_dos_casos(casos: pd.DataFrame, controles: pd.DataFrame) -> pd.DataFrame:
    """A linha do controle de cada caso, na ordem dos casos (o pareamento ja foi conferido nas visoes)."""
    por_id = controles.set_index(controles["variant_id"].astype(str), drop=False)
    return por_id.loc[casos["matched_variant_id"].astype(str).to_numpy()].reset_index(drop=True)


def filtrar_pares(casos: pd.DataFrame, controles: pd.DataFrame, manter: np.ndarray) -> tuple[pd.DataFrame,
                                                                                            pd.DataFrame]:
    """Mantem os pares marcados SEM desfazer pares: o controle sai e fica junto com o seu caso. Selecionar casos e
    controles separadamente descasaria os pares que o Mosaic materializou."""
    manter = np.asarray(manter, dtype=bool)
    if manter.shape != (len(casos),):
        raise ValueError("uma marca por par (na ordem dos casos)")
    casos_mantidos = casos[manter].reset_index(drop=True)
    parceiros = set(casos_mantidos["matched_variant_id"].astype(str))
    return casos_mantidos, controles[controles["variant_id"].astype(str).isin(parceiros)].reset_index(drop=True)


def _sensibilidade(nome: str, natureza: str, casos: pd.DataFrame, controles: pd.DataFrame, manter: np.ndarray,
                   pontos: dict[str, pd.Series], *, replicas: int, seed: int) -> dict[str, Any]:
    casos_mantidos, controles_mantidos = filtrar_pares(casos, controles, manter)
    return {"analise": nome, "natureza": natureza, "pares_mantidos": int(len(casos_mantidos)),
            "pares_retirados": int(len(casos) - len(casos_mantidos)),
            "composicao": {CASOS_PAREADOS: composicao(casos_mantidos), CONTROLES: composicao(controles_mantidos)},
            "interacao": (interacao(casos_mantidos, controles_mantidos, pontos, replicas=replicas, seed=seed)
                          if len(casos_mantidos) else None)}


def _nao_calculada(nome: str, motivo: str) -> dict[str, Any]:
    """Uma analise declarada nao some por falta de entrada: fica registrada, com o motivo."""
    return {"analise": nome, "nao_calculada": motivo}


def sensibilidades(estudo: str, casos: pd.DataFrame, controles: pd.DataFrame, pontos: dict[str, pd.Series], *,
                   replicas: int, seed: int, controles_com_scv_brasileira: set[str] | None,
                   exposicao: pd.Series | None, tolerancia_de_exposicao: int) -> dict[str, Any]:
    """As analises secundarias pre-declaradas da interacao. Todas mantem pares inteiros; nenhuma muda o resultado
    oficial; nenhuma e escolhida depois dos resultados -- e entrada ausente vira `nao_calculada`, nao silencio."""
    parceiros = controles_dos_casos(casos, controles)
    ids_dos_controles = parceiros["variant_id"].astype(str)
    saida: dict[str, Any] = {}
    kwargs = {"pontos": pontos, "replicas": replicas, "seed": seed}

    if estudo == ESTUDO_CLINICO:
        manter = (~casos["present_abraom"].astype(bool).to_numpy()
                  & ~parceiros["present_abraom"].astype(bool).to_numpy())
        saida["pares_ambos_fora_do_abraom"] = _sensibilidade(
            "pares com caso E controle fora do ABraOM",
            "PROPOSTA (plano, achado de 20/09: casos 4,6x mais presentes no ABraOM); se o efeito sumir, nao prova que "
            "era o banco (menos poder, outra composicao); se persistir, nao elimina todo efeito de contexto",
            casos, controles, manter, **kwargs)
        if controles_com_scv_brasileira is None:
            saida["sem_controles_com_scv_brasileira"] = _nao_calculada(
                "sem os pares cujo controle tem SCV brasileira", "lista da regra ampla nao fornecida")
        else:
            manter = ~ids_dos_controles.isin(set(map(str, controles_com_scv_brasileira))).to_numpy()
            saida["sem_controles_com_scv_brasileira"] = _sensibilidade(
                "sem os pares cujo controle tem SCV de instituicao brasileira (regra ampla)",
                "pre-declarada (plano 6.4): sem refazer o pareamento; a direcao de um eventual efeito nao e assumida",
                casos, controles, manter, **kwargs)
    else:
        motivo = "o estudo populacional e definido pela presenca no ABraOM e a analise declarada e do clinico"
        saida["pares_ambos_fora_do_abraom"] = _nao_calculada("pares com caso E controle fora do ABraOM", motivo)
        saida["sem_controles_com_scv_brasileira"] = _nao_calculada("sem os pares cujo controle tem SCV brasileira",
                                                                   "declarada so para o estudo clinico (plano 6.4)")

    if exposicao is None:
        saida["exposicao_empatada"] = _nao_calculada("pares com exposicao de locus equilibrada",
                                                     "exposicao por membro nao fornecida")
    else:
        ids_dos_casos = casos["variant_id"].astype(str)
        faltando = sorted((set(ids_dos_casos) | set(ids_dos_controles)) - set(exposicao.index.astype(str)))
        if faltando:
            raise EstudoInvalido(f"{len(faltando)} membros sem exposicao medida, ex.: {faltando[:3]}")
        diferenca = np.abs(exposicao.reindex(ids_dos_casos).to_numpy(dtype=float)
                           - exposicao.reindex(ids_dos_controles).to_numpy(dtype=float))
        saida["exposicao_empatada"] = _sensibilidade(
            f"pares com diferenca de exposicao de locus <= {tolerancia_de_exposicao} variante(s) de treino na janela",
            "pre-declarada (plano 6.4): exposicao igual nao implica efeito igual nos dois sistemas",
            casos, controles, diferenca <= tolerancia_de_exposicao, **kwargs)

    a_casos, a_controles = _arrays(casos, pontos), _arrays(parceiros, pontos)
    completos = (np.isfinite(a_casos[BASE]) & np.isfinite(a_casos[REGIONALIZADO])
                 & np.isfinite(a_controles[BASE]) & np.isfinite(a_controles[REGIONALIZADO]))
    if completos.all():
        saida["pares_completos_na_cobertura"] = {"analise": "so os pares com os dois membros cobertos",
                                                 "nao_necessaria": "todo par tem os dois membros cobertos"}
    else:
        saida["pares_completos_na_cobertura"] = _sensibilidade(
            "so os pares com os dois membros cobertos pelos dois sistemas",
            "pre-declarada: com cobertura incompleta, a intersecao de cada grupo desfaz a correspondencia entre os "
            "grupos", casos, controles, completos, **kwargs)
    return saida


def avaliar_estudo(membros: pd.DataFrame, estudo: str, pontos: dict[str, pd.Series], *,
                   replicas: int = REPLICAS, seed: int = SEED,
                   limiares: dict[str, float] | None = None,
                   controles_com_scv_brasileira: set[str] | None = None,
                   exposicao: pd.Series | None = None, tolerancia_de_exposicao: int = 0,
                   com_brier: bool = False,
                   progresso: Callable[[str], None] | None = None) -> dict[str, Any]:
    """Tudo o que o protocolo pede de UM estudo, mais as analises secundarias declaradas para ele.

    `pontos[sistema]` e uma Series indexada por `variant_id` com a probabilidade do sistema congelado; variante
    ausente ou NaN conta como nao pontuada (cobertura), nunca e imputada. `controles_com_scv_brasileira` e a lista
    da regra ampla (G2); `exposicao` e o `n_janela` por membro no snapshot final, com o raio declarado no G6.
    `com_brier` exige probabilidades em [0, 1]. Os tres coortes saem com as celulas por painel e sem cada painel:
    as margens podem pedir painel em qualquer um deles.
    """
    avisar = progresso or (lambda _texto: None)
    for sistema in SISTEMAS:
        if sistema not in pontos:
            raise EstudoInvalido(f"sem scores do sistema {sistema!r}")
        if pontos[sistema].index.has_duplicates:
            raise EstudoInvalido(f"scores do sistema {sistema!r} com variant_id repetido")
        valores = pontos[sistema].to_numpy(dtype=float)
        if com_brier and ((valores[np.isfinite(valores)] < 0) | (valores[np.isfinite(valores)] > 1)).any():
            raise EstudoInvalido(f"Brier so sobre probabilidades: {sistema!r} tem score fora de [0, 1]")
    v = visoes(membros, estudo)
    completo, casos, controles = v[COORTE_COMPLETO], v[CASOS_PAREADOS], v[CONTROLES]
    saida: dict[str, Any] = {
        "estudo": estudo,
        "pareamento": {"casos": int(len(completo)), "pareados": int(len(casos)),
                       "sem_par": int(len(completo) - len(casos)),
                       "taxa_de_pareamento": len(casos) / len(completo) if len(completo) else None},
        "metricas_com_limiar": ("com o limiar do ensemble congelado no G6" if limiares is not None
                                else "omitidas: sem limiar externo congelado"),
        "brier": ("sobre as probabilidades calibradas do ensemble (cada cabeca calibrada no fold 1 do core_locus); "
                  "delta regionalized - base NEGATIVO e melhora; a prevalencia do estudo difere da do fold 1 e pesa no "
                  "valor absoluto, nao na comparacao entre os sistemas no mesmo coorte" if com_brier
                  else "nao calculado"),
        "coortes": {},
    }
    avisar(f"{estudo}: coorte completo")
    saida["coortes"][COORTE_COMPLETO] = analise_do_coorte(completo, pontos, replicas=replicas, seed=seed,
                                                          limiares=limiares, por_painel=True, com_brier=com_brier)
    for nome, frame in ((CASOS_PAREADOS, casos), (CONTROLES, controles)):
        avisar(f"{estudo}: {nome}")
        saida["coortes"][nome] = analise_do_coorte(frame, pontos, replicas=replicas, seed=seed, limiares=limiares,
                                                   por_painel=True, com_brier=com_brier)
    avisar(f"{estudo}: interacao")
    saida["interacao"] = interacao(casos, controles, pontos, replicas=replicas, seed=seed)

    if estudo == ESTUDO_CLINICO:
        presentes = completo[completo["present_abraom"].astype(bool)].reset_index(drop=True)
        avisar(f"{estudo}: subconjunto present_abraom")
        saida["subconjuntos"] = {
            "present_abraom": {
                "exigido_por": "Mosaic (source_overlap_by_study: report_present_abraom_overlap)",
                "coorte": "coorte completo com present_abraom = true",
                **analise_do_coorte(presentes, pontos, replicas=replicas, seed=seed, limiares=limiares,
                                    com_brier=com_brier)},
        }
    avisar(f"{estudo}: sensibilidades")
    saida["sensibilidades"] = sensibilidades(estudo, casos, controles, pontos, replicas=replicas, seed=seed,
                                             controles_com_scv_brasileira=controles_com_scv_brasileira,
                                             exposicao=exposicao, tolerancia_de_exposicao=tolerancia_de_exposicao)
    return saida


# ------------------------------------------------------------------------------------------------ baselines

def _um_score(frame: pd.DataFrame, score: pd.Series) -> dict[str, np.ndarray]:
    ids = frame["variant_id"].astype(str)
    return {"y": frame["binary_label"].astype(int).to_numpy(), "s": score.reindex(ids).to_numpy(dtype=float),
            "clusters": frame["overlap_cluster_id"].astype(str).to_numpy(),
            "paineis": frame["primary_panel"].astype(str).to_numpy()}


def _metricas_de_um_score(y: np.ndarray, s: np.ndarray) -> dict[str, Any]:
    """AUROC e AUPRC na cobertura do proprio score. Score constante na cobertura nao discrimina por construcao."""
    coberto = np.isfinite(s)
    if np.unique(s[coberto]).size <= 1:
        return {"auroc": None, "auprc": None, "constante": True}
    return {**continuas(y[coberto], s[coberto]), "constante": False}


def avaliar_score_unico(frame: pd.DataFrame, score: pd.Series, *, replicas: int = REPLICAS,
                        seed: int = SEED) -> dict[str, Any]:
    """Um score FORA dos sistemas num coorte: suporte, AUROC e AUPRC na propria cobertura (coorte e paineis de
    discriminacao), com IC por cluster. Sem Brier: uma baseline e score de ordenacao, nao probabilidade."""
    a = _um_score(frame, score)
    celulas = {"coorte": np.ones(len(frame), dtype=bool)}
    for painel in PAINEIS_DE_DISCRIMINACAO:
        if (a["paineis"] == painel).any():
            celulas[f"painel:{painel}"] = a["paineis"] == painel

    def medir(indices: np.ndarray) -> dict[str, dict[str, Any]]:
        return {nome: _metricas_de_um_score(a["y"][indices[m[indices]]], a["s"][indices[m[indices]]])
                for nome, m in celulas.items()}

    observado = medir(np.arange(len(frame)))
    acumulador, rng = _Acumulador(), np.random.default_rng(seed)
    todas = grupos(a["clusters"])
    for _ in range(replicas):
        for nome, resultado in medir(sortear(todas, rng)).items():
            for chave in CONTINUAS:
                acumulador.guardar((nome, chave), resultado[chave])
    saida: dict[str, Any] = {"composicao": composicao(frame)}
    for nome, resultado in observado.items():
        saida[nome] = {"suporte": suporte(a["y"][celulas[nome]], np.isfinite(a["s"][celulas[nome]])),
                       "constante": resultado["constante"],
                       **{chave: {"estimativa": resultado[chave], **acumulador.intervalo((nome, chave))}
                          for chave in CONTINUAS}}
        if resultado["constante"]:
            saida[nome]["motivo"] = "score constante na cobertura: nao discrimina por construcao"
    return saida


def diferenca_entre_grupos(casos: pd.DataFrame, controles: pd.DataFrame, score: pd.Series, *,
                           replicas: int = REPLICAS, seed: int = SEED) -> dict[str, Any]:
    """AUROC/AUPRC do MESMO score nos casos pareados menos nos controles, com clusters sorteados em conjunto sobre a
    uniao. DESCRITIVA: diz se a baseline discrimina diferente nos dois grupos -- contexto para ler a interacao dos
    sistemas, nao criterio."""
    ac, ak = _um_score(casos, score), _um_score(controles, score)

    def estimar(ic: np.ndarray, ik: np.ndarray) -> dict[str, float | None]:
        mc, mk = _metricas_de_um_score(ac["y"][ic], ac["s"][ic]), _metricas_de_um_score(ak["y"][ik], ak["s"][ik])
        return {k: _delta(mk[k], mc[k]) for k in CONTINUAS}

    observado = estimar(np.arange(len(casos)), np.arange(len(controles)))
    todas = grupos(np.concatenate([ac["clusters"], ak["clusters"]]))
    acumulador, rng = _Acumulador(), np.random.default_rng(seed)
    for _ in range(replicas):
        indices = sortear(todas, rng)
        ic, ik = indices[indices < len(casos)], indices[indices >= len(casos)] - len(casos)
        replica = estimar(ic, ik) if ic.size and ik.size else {}
        for k in CONTINUAS:
            acumulador.guardar((k,), replica.get(k))
    return {"definicao": "metrica nos casos pareados - metrica nos controles, cada uma na cobertura do score",
            "natureza": "descritiva",
            **{k: {"estimativa": observado[k], **acumulador.intervalo((k,))} for k in CONTINUAS}}


def avaliar_baseline(membros: pd.DataFrame, estudo: str, score: pd.Series, *, especificacao: dict[str, Any],
                     replicas: int = REPLICAS, seed: int = SEED) -> dict[str, Any]:
    """Uma baseline declarada num estudo: regra, papel e cobertura explicitos; nos tres coortes, metricas absolutas;
    e a diferenca descritiva entre casos pareados e controles. Estudo em que a baseline nao se aplica por construcao
    sai com o motivo declarado, sem metrica."""
    base = {"regra": especificacao["regra"], "papel": especificacao["papel"].get(estudo),
            "sem_brier": "score de ordenacao, nao probabilidade"}
    if estudo in especificacao.get("nao_aplicavel", {}):
        return {**base, "nao_aplicavel": especificacao["nao_aplicavel"][estudo]}
    if score.index.has_duplicates:
        raise EstudoInvalido(f"baseline {especificacao['nome']} com variant_id repetido")
    v = visoes(membros, estudo)
    return {**base,
            "coortes": {nome: avaliar_score_unico(frame, score, replicas=replicas, seed=seed)
                        for nome, frame in v.items()},
            "diferenca_casos_pareados_menos_controles": diferenca_entre_grupos(
                v[CASOS_PAREADOS], v[CONTROLES], score, replicas=replicas, seed=seed)}
