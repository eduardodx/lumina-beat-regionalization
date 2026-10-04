#!/usr/bin/env python3
"""Fase 1, passo 3: os seis bracos sobre o R03 congelado, nas cinco execucoes da vista de 4 kb do Mosaic v1.

Bracos (docs/fase1_r03_congelado.md §1), todos com a mesma cabeca e as mesmas linhas:
- F: bloco de frequencia global oficial do Mosaic (`frequency_arms.frequency_features`), replicado aqui e conferido
  contra o oficial, valor a valor, antes de treinar;
- F+BR: F + bloco ABraOM (log10 AF, estados present/ac0/no_call/not_found, FILTER PASS, AN/2.342);
- E: `leitura_antiga_1344` do R03 congelado, dos caches do passo 2;
- E+F e E+F+BR;
- S+F: score do braco E + F. O S das linhas de treino e fora da amostra: cross-fitting pelos tres folds de treino da
  execucao, com o C escolhido para E e purga interna de 4.096 bp. Na validation e no teste, S e o score do modelo E
  da execucao. O `build_frequency_arms.py` oficial nao serve aqui: para um sistema treinado ele toma um score por
  variante sem respeitar a execucao.

Cabeca: regressao logistica L2 (sklearn, newton-cholesky) sobre features padronizadas no treino; C da grade
{1e-3, 1e-2, 1e-1, 1, 10} pela macro AUROC de missense, splice e noncoding na validation; empate fica com o menor C.

Linhas da execucao i (as mesmas em todos os bracos; so variantes elegiveis em 4 kb):
- treino: folds fora de {i, i+1}, gold + consensus, sem as purgas do run i;
- validation: fold i+1, gold, sem as purgas (onde se escolhe C; o avaliador do Mosaic escolhe o limiar ali);
- teste: fold i, todos os tiers.

USO (notebook, no .venv do Mosaic: sklearn e o pacote `mosaic` para as conferencias)
    PYTHONPATH="$PWD" uv run --project ~/mosaic-v1-2026-09-30 --frozen python scripts/fase1_bracos.py \\
        --entrega ~/mosaic-v1-2026-09-30 --cache ~/artifacts/redesenho/g3_cache/M0 \\
        --cache ~/artifacts/redesenho/g7_cache/M0 --cache ~/artifacts/mosaic_v1/cache_M0_complemento \\
        --out-dir ~/artifacts/mosaic_v1/bracos

SAIDAS (em --out-dir, que nao pode existir; tudo e gravado em <out-dir>.tmp e so renomeado no fim, depois das
checagens de contrato do proprio Mosaic):
- <id>/predictions.parquet e <id>/system.yaml de cada braco, no contrato de candidato do Mosaic;
- modelos/<id>_run<i>.npz: coeficientes, intercepto, media, desvio, colunas e C (para pontuar fora do release);
- selecao.json: macro da validation por C, C escolhido, iteracoes, tempo e avisos de cada ajuste;
- linhas.json: contagens e hashes das linhas por execucao e papel, estados do ABraOM e do gnomAD;
- fontes.json: identidade do release, hashes dos arquivos, dos caches e do script, revisao e versoes;
- diagnosticos/s_fora_da_amostra.parquet: o S das linhas de treino do braco S+F.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from eval.campanha import metricas  # noqa: E402
from eval.campanha.cache import indices_existentes, nome_do_fragmento, sha256_do_arquivo  # noqa: E402
from eval.campanha.layout import EXTRACOES  # noqa: E402
from scripts.extrair_mosaic_v1 import tabela_do_release, validar_cache_antigo  # noqa: E402
from scripts.inventario_mosaic_v1 import CAMPOS_DA_IDENTIDADE, K, papeis  # noqa: E402

RELEASE = "artifacts/mosaic-v1-2026-09-30"
VISTA = "views/4kb/partitions.parquet"
PROTOCOLO_DE_ESTUDOS = "config/study-protocol.yaml"
JANELA_BP = 4096
EXTRACAO = "leitura_antiga_1344"
GRADE_C = (1e-3, 1e-2, 1e-1, 1.0, 10.0)
AN_MAXIMO_DO_ABRAOM = 2342
ESTADOS_DO_ABRAOM = ("present", "ac0", "no_call", "not_found")
COLUNAS_DO_GNOMAD = ("gnomad_status", "gnomad_v4_af", "gnomad_v4_popmax_af", "gnomad_v4_af_amr", "gnomad_v4_af_afr",
                     "gnomad_v4_af_nfe")
COLUNAS_DO_ABRAOM = ("abraom_af", "abraom_an", "abraom_status", "abraom_filter", "present_abraom")
#: A entrega consumida (GUIA_DE_SUBMISSAO.md §2.1). `main` confere contra o release e o protocolo em disco.
REFERENCIA_DO_BENCHMARK = {
    "release_identity_hash": "d93125804e7cbc06d1187fd51580eebb758c81871ec381c29fc0a345644e0dc1",
    "protocol_hash": "67470b1443c4bf1bb8abee2760508bf93a1c97a50f9e99b17056d4326ae43885",
    "study_protocol_sha256": "ab7325f8f509c169c70ab7655d03cdadc923bc02154fdb3d42a749068fe2c875",
}
#: GUIA_OPERACIONAL_DE_SPLITS.md §10.1, run 0, antes do filtro de sequencia (o teste gold ja e elegivel).
CONTAGENS_DO_GUIA_RUN0: dict[str, int] | None = {"treino": 194_666, "validation": 2_106, "teste_gold": 2_110}
PASSO = 32_768  # linhas por bloco ao montar e padronizar as matrizes grandes, para nao duplica-las na memoria
RECEITA: dict[str, Any] = {
    "modelo": "regressao logistica L2 (sklearn LogisticRegression, intercepto sem penalidade)",
    # max_iter folgado: com Hessiana mal condicionada o sklearn troca para lbfgs com as iteracoes que sobram.
    "solver": "newton-cholesky", "max_iter": 500, "tol": 1e-6,
    "grade_C": [str(c) for c in GRADE_C],
    "warm_start": "C em ordem crescente; o otimo de cada C e unico (L2 estritamente convexa)",
    "padronizacao": "media e desvio (ddof 0) das linhas de treino da execucao; desvio < 1e-8 vira 1",
    "selecao": "macro AUROC nao ponderada de missense, splice e noncoding na validation; empate fica com o menor C",
    "score": "decision function (logit); maior = mais patogenico",
}
#: Ordem de execucao: E vem antes de S+F, que usa o C e os scores de E da mesma execucao.
BRACOS: dict[str, dict[str, Any]] = {
    "F": {"id": "fase1-f", "blocos": ("f",), "context_bp": 1},
    "F+BR": {"id": "fase1-f-br", "blocos": ("f", "br"), "context_bp": 1},
    "E": {"id": "fase1-e", "blocos": ("e",), "context_bp": JANELA_BP},
    "E+F": {"id": "fase1-e-f", "blocos": ("e", "f"), "context_bp": JANELA_BP},
    "E+F+BR": {"id": "fase1-e-f-br", "blocos": ("e", "f", "br"), "context_bp": JANELA_BP},
    "S+F": {"id": "fase1-s-f", "blocos": ("s", "f"), "context_bp": JANELA_BP},
}

Ajustador = Callable[[np.ndarray, np.ndarray, tuple[float, ...]], list[dict[str, Any]]]


class FalhaDaFase1(ValueError):
    """Uma conferencia de entrada, de selecao ou de contrato falhou; o resultado nao e gravado como valido."""


@dataclass
class Dados:
    """As variantes elegiveis em 4 kb, na ordem do release, e os indices de cada papel por execucao."""
    ids: np.ndarray
    y: np.ndarray
    paineis: np.ndarray
    cromossomos: np.ndarray
    posicoes: np.ndarray
    folds: np.ndarray
    blocos: dict[str, np.ndarray]
    colunas: dict[str, list[str]]
    papeis: list[dict[str, np.ndarray]]


# ----------------------------------------------------------------------------------------------- release e features

def carregar_release(raiz: Path) -> pd.DataFrame:
    """Uma linha por variante do release: rotulo, coordenadas, vista de 4 kb, painel, gnomAD e ABraOM."""
    exemplos = pd.read_parquet(raiz / "clinical-variants.parquet",
                               columns=["variant_id", "chrom", "pos_1based", "ref", "alt", "binary_label", "label_tier"])
    vista = pd.read_parquet(raiz / VISTA, columns=["variant_id", "sequence_eligible", "overlap_cluster_id",
                                                   "core_fold", "core_purged_runs"])
    paineis_ = pd.read_parquet(raiz / "evaluation-panels.parquet", columns=["variant_id", "primary_panel"])
    anotacoes = pd.read_parquet(raiz / "variant-annotations.parquet",
                                columns=["variant_id", *COLUNAS_DO_GNOMAD, *COLUNAS_DO_ABRAOM])
    df = (exemplos.merge(vista, on="variant_id", validate="one_to_one")
          .merge(paineis_, on="variant_id", validate="one_to_one")
          .merge(anotacoes, on="variant_id", validate="one_to_one"))
    if len(df) != len(exemplos):
        raise FalhaDaFase1(f"o release perdeu linhas no join: {len(df)} de {len(exemplos)}")
    df["sequence_eligible"] = df["sequence_eligible"].astype(bool)
    return df


def log_af(valores: pd.Series) -> pd.Series:
    """O `log_af` do Mosaic: ausente vira 0, negativo vira 0, piso de 1e-6."""
    return np.log10(valores.fillna(0).clip(lower=0) + 1e-6)


def bloco_f(frame: pd.DataFrame) -> pd.DataFrame:
    """Replica `frequency_arms.frequency_features` sem ABraOM; `main` confere contra o oficial antes de treinar."""
    estado = frame["gnomad_status"].fillna("unknown")
    af_global = frame["gnomad_v4_af"].where(~estado.isin(["not_found", "ac0"]), 0)
    return pd.DataFrame({
        "global": log_af(af_global), "popmax": log_af(frame["gnomad_v4_popmax_af"]),
        "amr": log_af(frame["gnomad_v4_af_amr"]), "afr": log_af(frame["gnomad_v4_af_afr"]),
        "nfe": log_af(frame["gnomad_v4_af_nfe"]),
        "not_found": (estado == "not_found").astype(float), "ac0": (estado == "ac0").astype(float),
    }, index=frame.index)


def bloco_br(frame: pd.DataFrame) -> pd.DataFrame:
    """ABraOM: log10 AF (a mesma conta do oficial), um indicador por estado, FILTER PASS e AN / 2.342.

    Estado sem medicao nao vira AF zero para quem le: o valor imputado (-6 no log, 0 no AN) so existe para o
    classificador, e os indicadores dizem o que aconteceu. Estado nulo fica com os quatro indicadores em zero.
    """
    estado = frame["abraom_status"].fillna("").astype(str)
    estranhos = sorted(set(estado.unique()) - set(ESTADOS_DO_ABRAOM) - {""})
    if estranhos:
        raise FalhaDaFase1(f"abraom_status fora de {ESTADOS_DO_ABRAOM}: {estranhos}")
    an = pd.to_numeric(frame["abraom_an"]).fillna(0).astype(float)
    if (an < 0).any() or (an > AN_MAXIMO_DO_ABRAOM).any():
        raise FalhaDaFase1(f"abraom_an fora de [0, {AN_MAXIMO_DO_ABRAOM}]: max {an.max()}")
    saida: dict[str, pd.Series] = {"br_log_af": log_af(frame["abraom_af"])}
    for nome in ESTADOS_DO_ABRAOM:
        saida[f"br_{nome}"] = (estado == nome).astype(float)
    saida["br_pass"] = (frame["abraom_filter"].fillna("").astype(str) == "PASS").astype(float)
    saida["br_an"] = an / AN_MAXIMO_DO_ABRAOM
    return pd.DataFrame(saida, index=frame.index)


def conferir_frequencia_oficial(frame: pd.DataFrame, oficial: Callable[..., pd.DataFrame]) -> dict[str, float]:
    """F tem de ser IGUAL ao bloco oficial; o log AF e a presenca do BR, iguais as colunas oficiais do ABraOM."""
    nosso, deles = bloco_f(frame), oficial(frame)
    if list(deles.columns) != list(nosso.columns):
        raise FalhaDaFase1(f"colunas do bloco F oficial mudaram: {list(deles.columns)}")
    deles_br, br = oficial(frame, with_abraom=True), bloco_br(frame)

    def diferenca(a: Any, b: Any) -> float:
        a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
        return float(np.max(np.abs(a - b))) if a.size else 0.0

    saida = {"f": diferenca(nosso, deles), "br_log_af": diferenca(br["br_log_af"], deles_br["abraom"]),
             "br_present": diferenca(br["br_present"], deles_br["abraom_present"])}
    if any(v != 0 for v in saida.values()):   # NaN tambem reprova
        raise FalhaDaFase1(f"bloco de frequencia difere do oficial do Mosaic: {saida}")
    return saida


def linhas_por_execucao(release: pd.DataFrame) -> tuple[list[dict[str, np.ndarray]], list[dict[str, Any]]]:
    """Indices (na tabela das elegiveis) de treino, validation e teste de cada execucao, pelos papeis do inventario."""
    elegivel = release["sequence_eligible"].to_numpy(dtype=bool)
    posicao = np.cumsum(elegivel) - 1
    por_run, contagens = [], []
    vezes_no_teste = np.zeros(int(elegivel.sum()), dtype=int)
    for run in range(K):
        m = papeis(release, run)
        idx = {"treino": posicao[m["treino"] & elegivel], "validation": posicao[m["validation"] & elegivel],
               "teste": posicao[m["teste_todos"] & elegivel]}
        vezes_no_teste[idx["teste"]] += 1
        por_run.append(idx)
        contagens.append({"run": run,
                          "guia": {"treino": int(m["treino"].sum()), "validation": int(m["validation"].sum()),
                                   "teste_gold": int(m["teste_gold"].sum())},
                          "efetivas": {nome: int(len(v)) for nome, v in idx.items()},
                          "fora_da_elegibilidade": {"treino": int((m["treino"] & ~elegivel).sum()),
                                                    "validation": int((m["validation"] & ~elegivel).sum())}})
    if (vezes_no_teste != 1).any():
        raise FalhaDaFase1("toda variante elegivel tem de ser teste em exatamente uma execucao")
    return por_run, contagens


def carregar_leitura(caches: list[Path], raiz: Path, ids: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """A `leitura_antiga_1344` de cada variante elegivel, alinhada a `ids`, da uniao dos caches do passo 2.

    Cada cache passa pela validacao do passo 2 (`validar_cache_antigo`) contra a identidade do primeiro: mesma
    identidade numerica, manifesto completo, hashes da tabela, fragmentos e coordenadas no release. Variante em mais
    de um cache, variante elegivel sem leitura ou valor nao finito reprovam.
    """
    if not caches:
        raise FalhaDaFase1("declare os caches do passo 2 (--cache)")
    referencia = json.loads((caches[0] / "identidade.json").read_text(encoding="utf-8"))
    if referencia.get("sistema") != "M0" or referencia.get("adapter_sha256") is not None:
        raise FalhaDaFase1("o primeiro cache tem de ser do M0 (R03 congelado, sem adapter)")
    tabela = tabela_do_release(raiz)
    posicao = pd.Series(np.arange(len(ids), dtype=np.float64), index=pd.Index(ids))
    matriz = np.zeros((len(ids), EXTRACOES[EXTRACAO][1]), dtype=np.float32)
    preenchida = np.zeros(len(ids), dtype=bool)
    fontes: dict[str, Any] = {}
    for pasta in caches:
        try:
            _, fonte = validar_cache_antigo(pasta, referencia, tabela)
        except (ValueError, KeyError, OSError) as exc:
            raise FalhaDaFase1(str(exc)) from exc
        lidas = 0
        for indice in indices_existentes(pasta):
            with open(pasta / nome_do_fragmento(indice), "rb") as arquivo, np.load(arquivo, allow_pickle=False) as dados:
                alvo = posicao.reindex(dados["variant_id"].astype(str)).to_numpy()
                usar = ~np.isnan(alvo)
                if not usar.any():
                    continue
                linhas = alvo[usar].astype(np.int64)
                if preenchida[linhas].any():
                    raise FalhaDaFase1(f"{pasta}: variantes ja lidas de outro cache")
                matriz[linhas] = np.asarray(dados[EXTRACAO], dtype=np.float32)[usar]
                preenchida[linhas] = True
                lidas += int(usar.sum())
        identidade = json.loads((pasta / "identidade.json").read_text(encoding="utf-8"))
        fontes[str(pasta)] = {**fonte, "lidas_no_release": lidas,
                              "identidade": {campo: identidade.get(campo) for campo in CAMPOS_DA_IDENTIDADE}}
    if not preenchida.all():
        raise FalhaDaFase1(f"{int((~preenchida).sum())} variantes elegiveis sem {EXTRACAO} nos caches")
    if not np.isfinite(matriz).all():
        raise FalhaDaFase1(f"{EXTRACAO} com valores nao finitos")
    return matriz, fontes


def montar_dados(elegiveis: pd.DataFrame, matriz_e: np.ndarray, por_run: list[dict[str, np.ndarray]]) -> Dados:
    if elegiveis["binary_label"].isna().any():
        raise FalhaDaFase1("variante elegivel sem rotulo")
    f, br = bloco_f(elegiveis), bloco_br(elegiveis)
    return Dados(
        ids=elegiveis["variant_id"].astype(str).to_numpy(), y=elegiveis["binary_label"].astype(int).to_numpy(),
        paineis=elegiveis["primary_panel"].astype(str).to_numpy(),
        cromossomos=elegiveis["chrom"].astype(str).to_numpy(), posicoes=elegiveis["pos_1based"].to_numpy(np.int64),
        folds=elegiveis["core_fold"].to_numpy(np.int64),
        blocos={"e": matriz_e, "f": f.to_numpy(np.float64), "br": br.to_numpy(np.float64)},
        colunas={"e": [f"e_{i:04d}" for i in range(matriz_e.shape[1])], "f": list(f.columns), "br": list(br.columns),
                 "s": ["s_e"]},
        papeis=por_run)


# --------------------------------------------------------------------------------------------- matrizes e ajuste

def montar(dados: Dados, blocos: tuple[str, ...], indices: np.ndarray, s: np.ndarray | None = None) -> np.ndarray:
    """Matriz float64 das linhas `indices` com os blocos na ordem dada, montada por partes (sem copia inteira extra)."""
    larguras = [1 if b == "s" else dados.blocos[b].shape[1] for b in blocos]
    X = np.empty((len(indices), sum(larguras)), dtype=np.float64)
    inicio = 0
    for bloco, largura in zip(blocos, larguras):
        if bloco == "s":
            if s is None or len(s) != len(indices):
                raise FalhaDaFase1("o bloco s precisa de um score por linha")
            X[:, inicio] = s
        else:
            fonte = dados.blocos[bloco]
            for a in range(0, len(indices), PASSO):
                X[a:a + PASSO, inicio:inicio + largura] = fonte[indices[a:a + PASSO]]
        inicio += largura
    return X


def padronizador(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Media e desvio (ddof 0) por coluna, por partes; desvio < 1e-8 vira 1 (coluna constante no treino)."""
    if len(X) == 0:
        raise FalhaDaFase1("treino vazio")
    media = np.zeros(X.shape[1])
    for a in range(0, len(X), PASSO):
        media += X[a:a + PASSO].sum(axis=0)
    media /= len(X)
    quadrados = np.zeros(X.shape[1])
    for a in range(0, len(X), PASSO):
        d = X[a:a + PASSO] - media
        quadrados += np.einsum("ij,ij->j", d, d)
    desvio = np.sqrt(quadrados / len(X))
    desvio[desvio < 1e-8] = 1.0
    return media, desvio


def padronizar(X: np.ndarray, media: np.ndarray, desvio: np.ndarray) -> np.ndarray:
    for a in range(0, len(X), PASSO):
        X[a:a + PASSO] -= media
        X[a:a + PASSO] /= desvio
    return X


def ajustar_sklearn(Z: np.ndarray, y: np.ndarray, grade: tuple[float, ...]) -> list[dict[str, Any]]:
    """Um ajuste por C, em ordem crescente, partindo da solucao do C anterior. Registra iteracoes, tempo e avisos."""
    from sklearn.linear_model import LogisticRegression

    modelo = LogisticRegression(solver=RECEITA["solver"], max_iter=RECEITA["max_iter"], tol=RECEITA["tol"],
                                warm_start=True)
    saida = []
    for c in grade:
        modelo.set_params(C=float(c))
        inicio = time.perf_counter()
        with warnings.catch_warnings(record=True) as avisos:
            warnings.simplefilter("always")
            modelo.fit(Z, y)
        textos = sorted({f"{a.category.__name__}: {a.message}" for a in avisos})
        n_iter = int(np.max(modelo.n_iter_))
        saida.append({"C": float(c), "coef": np.array(modelo.coef_[0], dtype=np.float64),
                      "intercepto": float(modelo.intercept_[0]), "n_iter": n_iter,
                      "segundos": round(time.perf_counter() - inicio, 3), "avisos": textos,
                      "convergiu": n_iter < RECEITA["max_iter"] and not any(t.startswith("ConvergenceWarning")
                                                                           for t in textos)})
    return saida


#: O ajustador usado por `main`. Os testes trocam por um substituto quando o sklearn nao existe (Windows).
AJUSTADOR: Ajustador = ajustar_sklearn


class CabecaSemSelecao(FalhaDaFase1):
    """Nenhum C teve a macro definida na validation."""


def escolher_c(grade: list[dict[str, Any]]) -> int:
    """Indice do C com a maior macro da validation; empate fica com o menor C (a grade e crescente)."""
    validos = [(r["macro_validacao"], -i) for i, r in enumerate(grade) if r["macro_validacao"] is not None]
    if not validos:
        raise CabecaSemSelecao("nenhum C com macro definida na validation")
    return -max(validos)[1]


def pontuar(Z: np.ndarray, coef: np.ndarray, intercepto: float) -> np.ndarray:
    return Z @ coef + intercepto


def treinar_braco(dados: Dados, run: int, blocos: tuple[str, ...], ajustador: Ajustador,
                  s: dict[str, np.ndarray] | None = None) -> dict[str, Any]:
    """Padroniza no treino, ajusta a grade, escolhe C na validation e pontua validation e teste com o escolhido."""
    idx = dados.papeis[run]
    y_tr = dados.y[idx["treino"]]
    if len(np.unique(y_tr)) < 2:
        raise FalhaDaFase1(f"run {run}: treino sem as duas classes")
    X = montar(dados, blocos, idx["treino"], s=None if s is None else s["treino"])
    media, desvio = padronizador(X)
    grade = ajustador(padronizar(X, media, desvio), y_tr, GRADE_C)
    del X
    Xva = padronizar(montar(dados, blocos, idx["validation"], s=None if s is None else s["validation"]), media, desvio)
    y_va, p_va = dados.y[idx["validation"]], dados.paineis[idx["validation"]]
    for r in grade:
        painel = metricas.por_painel(pontuar(Xva, r["coef"], r["intercepto"]), y_va, p_va)
        r["macro_validacao"] = metricas.macro(painel)
        r["auroc_validacao_por_painel"] = {nome: v["auroc"] for nome, v in painel.items()}
    escolhido = grade[escolher_c(grade)]
    Xte = padronizar(montar(dados, blocos, idx["teste"], s=None if s is None else s["teste"]), media, desvio)
    colunas = [c for b in blocos for c in dados.colunas[b]]
    return {"C": escolhido["C"], "macro_validacao": escolhido["macro_validacao"], "coef": escolhido["coef"],
            "intercepto": escolhido["intercepto"], "media": media, "desvio": desvio, "colunas": colunas,
            "score_validation": pontuar(Xva, escolhido["coef"], escolhido["intercepto"]),
            "score_teste": pontuar(Xte, escolhido["coef"], escolhido["intercepto"]),
            "grade": [{k: v for k, v in r.items() if k != "coef"} for r in grade]}


def perto_de(cromossomos: np.ndarray, posicoes: np.ndarray, referencia: np.ndarray, consulta: np.ndarray,
             janela: int = JANELA_BP) -> np.ndarray:
    """Se cada linha de `consulta` tem alguma de `referencia` no mesmo cromossomo a menos de `janela` bp: a regra de
    `mosaic.views._within` (|delta| <= janela - 1)."""
    saida = np.zeros(len(consulta), dtype=bool)
    crom_ref, pos_ref = cromossomos[referencia], posicoes[referencia]
    crom_q, pos_q = cromossomos[consulta], posicoes[consulta]
    for crom in np.unique(crom_q):
        alvo = np.sort(pos_ref[crom_ref == crom])
        if alvo.size == 0:
            continue
        m = crom_q == crom
        q = pos_q[m]
        saida[m] = (np.searchsorted(alvo, q + janela - 1, side="right")
                    > np.searchsorted(alvo, q - janela + 1, side="left"))
    return saida


def particao_interna(dados: Dados, run: int) -> list[dict[str, Any]]:
    """Os folds internos do treino da execucao: cada fold de treino vira o teste interno; o treino interno sao os
    outros folds de treino, menos as linhas a menos de 4.096 bp de alguma linha do teste interno."""
    treino = dados.papeis[run]["treino"]
    folds = dados.folds[treino]
    saida = []
    for fold in sorted(int(f) for f in np.unique(folds)):
        teste = treino[folds == fold]
        candidatos = treino[folds != fold]
        purgar = perto_de(dados.cromossomos, dados.posicoes, teste, candidatos)
        saida.append({"fold": fold, "teste": teste, "treino": candidatos[~purgar], "purgadas": int(purgar.sum())})
    return saida


def s_fora_da_amostra(dados: Dados, run: int, c: float, ajustador: Ajustador) -> tuple[np.ndarray, list[dict]]:
    """O score de E de cada linha de treino da execucao, por um modelo E (mesmo C) que nao a viu nem a vizinhos."""
    treino = dados.papeis[run]["treino"]
    if (np.diff(treino) <= 0).any():
        raise FalhaDaFase1("indices de treino fora de ordem")
    s = np.full(len(treino), np.nan)
    info = []
    for parte in particao_interna(dados, run):
        y_i = dados.y[parte["treino"]]
        if len(np.unique(y_i)) < 2:
            raise FalhaDaFase1(f"run {run}, fold interno {parte['fold']}: treino interno sem as duas classes")
        X = montar(dados, ("e",), parte["treino"])
        media, desvio = padronizador(X)
        ajuste = ajustador(padronizar(X, media, desvio), y_i, (c,))[0]
        del X
        Xt = padronizar(montar(dados, ("e",), parte["teste"]), media, desvio)
        s[np.searchsorted(treino, parte["teste"])] = pontuar(Xt, ajuste["coef"], ajuste["intercepto"])
        info.append({"fold_interno": parte["fold"], "treino_interno": int(len(parte["treino"])),
                     "teste_interno": int(len(parte["teste"])), "purgadas_internas": parte["purgadas"],
                     "C": float(c), "n_iter": ajuste["n_iter"], "segundos": ajuste["segundos"],
                     "avisos": ajuste["avisos"], "convergiu": ajuste["convergiu"]})
    if not np.isfinite(s).all():
        raise FalhaDaFase1(f"run {run}: S fora da amostra incompleto")
    return s, info


def _resumo(valores: np.ndarray) -> dict[str, float]:
    return {"media": float(np.mean(valores)), "desvio": float(np.std(valores)), "n": int(len(valores))}


def rodar_bracos(dados: Dados, ajustador: Ajustador,
                 registrar: Callable[[str], None] = lambda t: print(t, flush=True)) -> tuple[dict, pd.DataFrame]:
    """Os seis bracos nas cinco execucoes. Devolve os resultados por braco (lista por run) e o S do treino do S+F."""
    nomes = list(BRACOS)
    if nomes.index("E") > nomes.index("S+F"):
        raise FalhaDaFase1("E tem de rodar antes de S+F")
    resultados: dict[str, list[dict[str, Any]]] = {braco: [] for braco in BRACOS}
    s_treino: list[pd.DataFrame] = []
    for run in range(K):
        e_run: dict[str, Any] | None = None
        for braco, cfg in BRACOS.items():
            inicio = time.perf_counter()
            if braco == "S+F":
                s, info = s_fora_da_amostra(dados, run, e_run["C"], ajustador)
                treino = dados.papeis[run]["treino"]
                r = treinar_braco(dados, run, cfg["blocos"], ajustador,
                                  s={"treino": s, "validation": e_run["score_validation"], "teste": e_run["score_teste"]})
                r["s_interno"] = info
                r["s_resumo"] = {"treino_fora_da_amostra": _resumo(s), "validation": _resumo(e_run["score_validation"]),
                                 "teste": _resumo(e_run["score_teste"])}
                s_treino.append(pd.DataFrame({"variant_id": dados.ids[treino], "run": run,
                                              "fold_interno": dados.folds[treino], "score_e": s}))
            else:
                r = treinar_braco(dados, run, cfg["blocos"], ajustador)
            if braco == "E":
                e_run = r
            resultados[braco].append(r)
            iteracoes = ", ".join(f"{g['C']:g}:{g['n_iter']}" for g in r["grade"])
            registrar(f"[{braco} run {run}] C={r['C']:g} macro_validacao={r['macro_validacao']:.4f} "
                      f"(iteracoes {iteracoes}; {time.perf_counter() - inicio:.1f} s)")
    return resultados, pd.concat(s_treino, ignore_index=True)


# ------------------------------------------------------------------------------------------------------- saidas

def predicoes(dados: Dados, resultados_do_braco: list[dict[str, Any]]) -> pd.DataFrame:
    """Validation e teste de cada execucao no contrato do Mosaic (`core_locus`, janela 4.096)."""
    partes = []
    for run, r in enumerate(resultados_do_braco):
        for papel, chave, nome in (("validation", "score_validation", "validation"), ("test", "score_teste", "teste")):
            indices = dados.papeis[run][nome]
            partes.append(pd.DataFrame({"variant_id": dados.ids[indices], "study": "core_locus",
                                        "run": np.full(len(indices), run, dtype=np.int64), "role": papel,
                                        "window_bp": np.full(len(indices), JANELA_BP, dtype=np.int64),
                                        "score": np.asarray(r[chave], dtype=np.float64)}))
    saida = pd.concat(partes, ignore_index=True)
    if not np.isfinite(saida["score"].to_numpy()).all():
        raise FalhaDaFase1("score nao finito")
    return saida.sort_values(["run", "role", "variant_id"], kind="mergesort").reset_index(drop=True)


def sistema(braco: str, proveniencia: dict[str, Any]) -> dict[str, Any]:
    """O `system.yaml` do braco: contrato do Mosaic, exposicao declarada e a descricao da Fase 1."""
    cfg = BRACOS[braco]
    usa_r03 = bool({"e", "s"} & set(cfg["blocos"]))
    fontes_pop = []
    if "f" in cfg["blocos"]:
        fontes_pop.append("gnomAD v4.1 joint: bloco de frequencia oficial do Mosaic (frequency_features)")
    if "br" in cfg["blocos"]:
        fontes_pop.append("ABraOM WGS-1171 (variant-annotations): log AF, estado, FILTER PASS, AN/2342")
    rotulos = {"status": "used",
               "sources": ["mosaic-v1-2026-09-30 (ClinVar 2026-06): gold + consensus dos folds de treino de cada "
                           "execucao da vista de 4 kb; C escolhido na validation gold da execucao"]}
    populacao: dict[str, Any] = {"status": "used" if fontes_pop else "unknown", "sources": fontes_pop}
    if usa_r03:
        rotulos["backbone_r03"] = "desconhecida: exposicao do tronco R03 a rotulos clinicos nao conferida"
        populacao["backbone_r03"] = ("o R03 tem cabeca populacional com AF do gnomAD (alvo AF_joint segundo o plano "
                                     "do Mosaic; loss nao conferida)")
    return {
        "id": cfg["id"], "mode": "trained", "context_bp": cfg["context_bp"],
        "training_cutoff": None if usa_r03 else "2026-06",
        "exposure": {"clinical_labels": rotulos, "population_frequency": populacao},
        "benchmark": dict(REFERENCIA_DO_BENCHMARK),
        "fase1": {"braco": braco, "blocos": list(cfg["blocos"]),
                  "receita": {k: (repr(v) if isinstance(v, float) else v) for k, v in RECEITA.items()},
                  "linhas": ("treino: folds fora de {i, i+1}, gold + consensus, sem as purgas do run i; validation: "
                             "fold i+1, gold, sem as purgas; teste: fold i, todos os tiers; so elegiveis em 4 kb"),
                  "training_cutoff_nota": ("rotulos da cabeca: ClinVar 2026-06; data de treino do tronco R03 "
                                           "desconhecida" if usa_r03 else "rotulos da cabeca: ClinVar 2026-06"),
                  "s_mais_f": ("S nas linhas de treino: cross-fitting pelos tres folds de treino, com o C do braco E "
                               "e purga interna de 4.096 bp; na validation e no teste, o score do modelo E da "
                               "execucao") if braco == "S+F" else None,
                  "r03": proveniencia.get("r03") if usa_r03 else None,
                  "script_sha256": proveniencia.get("script_sha256"), "revisao": proveniencia.get("revisao")},
    }


def _floats(valor: Any, caminho: str = "") -> list[str]:
    if isinstance(valor, float):
        return [caminho or "<raiz>"]
    if isinstance(valor, dict):
        return [c for k, v in valor.items() for c in _floats(v, f"{caminho}.{k}")]
    if isinstance(valor, (list, tuple)):
        return [c for i, v in enumerate(valor) for c in _floats(v, f"{caminho}[{i}]")]
    return []


def escrever_sistema(pasta: Path, conteudo: dict[str, Any]) -> None:
    """JSON, que e YAML valido. Sem float: o YAML 1.1 do PyYAML le `1e-06` como texto, e o avaliador leria outra
    coisa. Se o PyYAML existir, confere ainda que o YAML le exatamente o que foi escrito."""
    com_float = _floats(conteudo)
    if com_float:
        raise FalhaDaFase1(f"system.yaml com float em {com_float}; grave como texto")
    texto = json.dumps(conteudo, indent=2, ensure_ascii=True) + "\n"
    (pasta / "system.yaml").write_text(texto, encoding="utf-8")
    try:
        import yaml
    except ImportError:
        return
    if yaml.safe_load(texto) != json.loads(texto):
        raise FalhaDaFase1(f"{pasta / 'system.yaml'}: o YAML nao rele o que o JSON escreveu")


def hash_dos_ids(ids: np.ndarray) -> str:
    return hashlib.sha256("\n".join(sorted(map(str, ids))).encode("utf-8")).hexdigest()


def relatorio_de_linhas(dados: Dados, elegiveis: pd.DataFrame, contagens: list[dict[str, Any]]) -> dict[str, Any]:
    por_run = []
    for run, idx in enumerate(dados.papeis):
        papeis_ = {}
        for nome, indices in idx.items():
            y = dados.y[indices]
            papeis_[nome] = {"n": int(len(indices)), "n_P": int((y == 1).sum()), "n_B": int((y == 0).sum()),
                             "por_painel": {p: int(n) for p, n in
                                            pd.Series(dados.paineis[indices]).value_counts().sort_index().items()},
                             "ids_sha256": hash_dos_ids(dados.ids[indices])}
        por_run.append({**contagens[run], "papeis": papeis_})
    estados = {c: {str(k): int(v) for k, v in elegiveis[c].fillna("<nulo>").value_counts().sort_index().items()}
               for c in ("abraom_status", "gnomad_status")}
    filtro = elegiveis.loc[elegiveis["abraom_status"] == "present", "abraom_filter"].fillna("<nulo>")
    return {"elegiveis": int(len(dados.ids)), "execucoes": por_run, "estados": estados,
            "abraom_filter_dos_presentes": {str(k): int(v) for k, v in filtro.value_counts().sort_index().items()}}


def revisao_do_repositorio() -> dict[str, Any]:
    try:
        commit = subprocess.run(["git", "-C", str(RAIZ), "rev-parse", "HEAD"], capture_output=True, text=True,
                                check=True).stdout.strip()
        modificados = subprocess.run(["git", "-C", str(RAIZ), "status", "--porcelain", "--", "scripts", "eval"],
                                     capture_output=True, text=True, check=True).stdout.splitlines()
        return {"commit": commit, "modificados": modificados}
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"commit": None, "erro": str(exc)}


def versoes() -> dict[str, Any]:
    saida = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__}
    try:
        import sklearn
        saida["sklearn"] = sklearn.__version__
    except ImportError:
        saida["sklearn"] = None
    return saida


def gravar(destino: Path, dados: Dados, resultados: dict[str, list[dict[str, Any]]], s_treino: pd.DataFrame,
           relatorios: dict[str, Any], proveniencia: dict[str, Any]) -> None:
    (destino / "modelos").mkdir(parents=True)
    (destino / "diagnosticos").mkdir()
    selecao: dict[str, Any] = {}
    for braco, cfg in BRACOS.items():
        pasta = destino / cfg["id"]
        pasta.mkdir()
        predicoes(dados, resultados[braco]).to_parquet(pasta / "predictions.parquet", index=False)
        escrever_sistema(pasta, sistema(braco, proveniencia))
        selecao[braco] = []
        for run, r in enumerate(resultados[braco]):
            np.savez(destino / "modelos" / f"{cfg['id']}_run{run}.npz", coef=r["coef"],
                     intercepto=np.float64(r["intercepto"]), media=r["media"], desvio=r["desvio"],
                     C=np.float64(r["C"]), colunas=np.array(r["colunas"]), braco=np.array(braco),
                     run=np.int64(run))
            selecao[braco].append({"run": run, "C": r["C"], "macro_validacao": r["macro_validacao"],
                                   "grade": r["grade"], **({"s_interno": r["s_interno"], "s_resumo": r["s_resumo"]}
                                                           if braco == "S+F" else {})})
    s_treino.to_parquet(destino / "diagnosticos" / "s_fora_da_amostra.parquet", index=False)
    (destino / "selecao.json").write_text(json.dumps({"receita": RECEITA, "bracos": selecao}, indent=2,
                                                     ensure_ascii=False), encoding="utf-8")
    (destino / "linhas.json").write_text(json.dumps(relatorios["linhas"], indent=2, ensure_ascii=False),
                                         encoding="utf-8")
    (destino / "fontes.json").write_text(json.dumps(proveniencia, indent=2, ensure_ascii=False, default=str),
                                         encoding="utf-8")


# ------------------------------------------------------------------------------------------------------------- main

def ferramentas_do_mosaic() -> dict[str, Callable[..., Any]]:
    """O que vem do pacote `mosaic` da entrega: a identidade do release, o bloco de frequencia oficial e as checagens
    do contrato de candidato. Sem elas nao ha conferencia; os testes substituem esta funcao."""
    import sklearn  # noqa: F401 -- falhar antes de ler os caches se o ambiente nao tiver o solver
    from mosaic.comparator_eval.candidate import (
        check_benchmark_reference,
        check_fold_roles,
        load_predictions,
        load_system,
    )
    from mosaic.comparator_eval.frequency_arms import frequency_features
    from mosaic.identity import verify_release_identity

    return {"verificar_identidade": verify_release_identity, "frequencia_oficial": frequency_features,
            "load_system": load_system, "load_predictions": load_predictions, "check_fold_roles": check_fold_roles,
            "check_benchmark_reference": check_benchmark_reference}


def conferir_entrega(identidade: dict[str, Any], protocolo_sha256: str) -> None:
    lidos = {"release_identity_hash": identidade.get("release_identity_hash"),
             "protocol_hash": identidade.get("protocol_hash"), "study_protocol_sha256": protocolo_sha256}
    diferentes = sorted(k for k, v in REFERENCIA_DO_BENCHMARK.items() if lidos[k] != v)
    if diferentes:
        raise FalhaDaFase1(f"a entrega nao e a esperada em {diferentes}: {lidos}")


def conferir_contrato(ferramentas: dict[str, Callable[..., Any]], pasta: Path, release: pd.DataFrame,
                      identidade: dict[str, Any], protocolo_sha256: str) -> int:
    """As checagens do proprio avaliador do Mosaic sobre o que foi gravado: system.yaml, referencia do benchmark,
    esquema das predicoes e folds coerentes com execucao e papel."""
    conteudo = ferramentas["load_system"](pasta / "system.yaml")
    ferramentas["check_benchmark_reference"](conteudo, identidade, protocol_sha256=protocolo_sha256)
    preds = ferramentas["load_predictions"](pasta / "predictions.parquet", conteudo)
    ferramentas["check_fold_roles"](release[["variant_id", "core_fold"]], preds, track="core_locus", window=JANELA_BP)
    return int(len(preds))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--entrega", required=True, type=Path, help="checkout dedicado do Mosaic com a entrega baixada")
    parser.add_argument("--cache", action="append", default=[], type=Path,
                        help="caches do passo 2; o primeiro fixa a identidade numerica")
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    entrega, destino = args.entrega.expanduser(), args.out_dir.expanduser()
    temporaria = destino.with_name(destino.name + ".tmp")
    if destino.exists() or temporaria.exists():
        return _falhar([f"{destino} ou {temporaria} ja existe; os bracos gravam sempre numa pasta nova"])
    try:
        ferramentas = ferramentas_do_mosaic()
    except ImportError as exc:
        return _falhar([f"sem o pacote mosaic ou o sklearn ({exc}); rodar no .venv do Mosaic (uv run --project)"])

    inicio = time.perf_counter()
    raiz = entrega / RELEASE
    try:
        identidade = ferramentas["verificar_identidade"](raiz, config_dir=entrega / "config")
        protocolo_sha256 = sha256_do_arquivo(entrega / PROTOCOLO_DE_ESTUDOS)
        conferir_entrega(identidade, protocolo_sha256)
        release = carregar_release(raiz)
        por_run, contagens = linhas_por_execucao(release)
        if CONTAGENS_DO_GUIA_RUN0 is not None and contagens[0]["guia"] != CONTAGENS_DO_GUIA_RUN0:
            raise FalhaDaFase1(f"run 0 nao bate com o guia de splits: {contagens[0]['guia']}")
        elegiveis = release[release["sequence_eligible"]].reset_index(drop=True)
        conferencia_f = conferir_frequencia_oficial(elegiveis, ferramentas["frequencia_oficial"])
        matriz_e, fontes_dos_caches = carregar_leitura([p.expanduser() for p in args.cache], raiz,
                                                       elegiveis["variant_id"].astype(str).to_numpy())
        dados = montar_dados(elegiveis, matriz_e, por_run)
    except (FalhaDaFase1, ValueError, KeyError, OSError, RuntimeError, AssertionError) as exc:
        return _falhar([f"{type(exc).__name__}: {exc}"])
    for c in contagens:
        print(f"run {c['run']}: guia {c['guia']} | efetivas {c['efetivas']} | fora da elegibilidade "
              f"{c['fora_da_elegibilidade']}", flush=True)
    print(f"bloco F igual ao oficial ({conferencia_f}); {len(dados.ids):,} elegiveis com {EXTRACAO} de "
          f"{len(fontes_dos_caches)} caches", flush=True)

    try:
        resultados, s_treino = rodar_bracos(dados, AJUSTADOR)
    except (FalhaDaFase1, ValueError) as exc:
        return _falhar([f"{type(exc).__name__}: {exc}"])

    primeira = next(iter(fontes_dos_caches.values()))["identidade"]
    proveniencia = {
        "entrega": str(entrega), "identidade_do_release": {k: identidade.get(k) for k in
                                                           ("release_id", "version", "release_identity_hash",
                                                            "protocol_hash")},
        "study_protocol_sha256": protocolo_sha256,
        "arquivos_do_release": {nome: sha256_do_arquivo(raiz / nome) for nome in
                                ("clinical-variants.parquet", VISTA, "evaluation-panels.parquet",
                                 "variant-annotations.parquet")},
        "conferencia_do_bloco_f": conferencia_f, "caches": fontes_dos_caches,
        "r03": {"extracao": EXTRACAO, "checkpoint_sha256": primeira.get("checkpoint_sha256"),
                "versao_do_extrator": primeira.get("versao_do_extrator"), "janela_bp": primeira.get("janela_bp")},
        "script_sha256": sha256_do_arquivo(Path(__file__)), "revisao": revisao_do_repositorio(), "versoes": versoes(),
        "receita": RECEITA, "segundos": round(time.perf_counter() - inicio, 1),
    }
    temporaria.mkdir(parents=True)
    try:
        gravar(temporaria, dados, resultados, s_treino,
               {"linhas": relatorio_de_linhas(dados, elegiveis, contagens)}, proveniencia)
        conferidas = {cfg["id"]: conferir_contrato(ferramentas, temporaria / cfg["id"], release, identidade,
                                                   protocolo_sha256) for cfg in BRACOS.values()}
    except (FalhaDaFase1, ValueError, KeyError, OSError, RuntimeError, AssertionError) as exc:
        return _falhar([f"{type(exc).__name__}: {exc} (saida parcial deixada em {temporaria} para inspecao)"])
    temporaria.rename(destino)

    print("\nmacro da validation (a mesma usada para escolher C; otimista por construcao):")
    for braco, rs in resultados.items():
        print(f"  {braco:7s} " + "  ".join(f"run{i} {r['macro_validacao']:.4f} (C={r['C']:g})" for i, r in enumerate(rs)))
    print(f"\nPASSOU: {len(conferidas)} bracos no contrato do Mosaic ({conferidas}) | {destino}")
    return 0


def _falhar(problemas: list[str]) -> int:
    print(f"FALHOU: {len(problemas)} problema(s)")
    for problema in problemas:
        print(f"  - {problema}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
