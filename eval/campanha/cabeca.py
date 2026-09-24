"""Cabeca clinica sobre o cache: a MESMA receita e o MESMO procedimento em M0 e em MR.

A receita e a do MLP da pesquisa de extracao (`mlp_scores` em `scripts/probe_feature_eval.py`, branch
`embedding-probe-mosaic`), portada sem mudar: Linear(d, 64) -> GELU -> Dropout(0,1) -> Linear(64, 1), Adam 3e-3 com
weight decay 1e-4, BCE em lote cheio, avaliacao a cada 10 epocas, paciencia de 10 avaliacoes, no maximo 600 epocas,
e a epoca escolhida pela macro-AUROC de missense/splice/noncoding na VALIDACAO (fold 1). Padronizacao ajustada so
no treino. Depois, Platt e limiar de MCC ajustados na mesma validacao (plano, secao 5.3).
"""
from __future__ import annotations

import copy
import math
from typing import Any

import numpy as np

from eval.campanha import metricas

RECEITA: dict[str, Any] = {"oculta": 64, "dropout": 0.1, "lr": 3e-3, "weight_decay": 1e-4, "max_epocas": 600,
                           "avaliar_a_cada": 10, "paciencia": 10,
                           "parada": "macro-AUROC de missense/splice/noncoding na validacao",
                           "padronizacao": "media e desvio do treino; desvio < 1e-8 vira 1",
                           "origem": "mlp_scores de scripts/probe_feature_eval.py (embedding-probe-mosaic)"}
#: Formato do arquivo que o comparador grava por cabeca (`cabeca_{sistema}_h{semente}.pt`).
FORMATO_DA_CABECA = "campanha_r03_cabeca_v1"


class CabecaSemSelecao(RuntimeError):
    """Nenhuma avaliacao teve a macro definida na validacao: nao ha epoca a escolher."""


def padronizador(treino: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    media = treino.mean(axis=0)
    desvio = treino.std(axis=0)
    desvio[desvio < 1e-8] = 1.0  # dimensao constante no treino: nao escala, nao explode
    return media, desvio


def montar_rede(dimensao: int, receita: dict[str, Any] = RECEITA) -> Any:
    """A arquitetura da cabeca. Uma so definicao para treinar e para recarregar o que foi salvo."""
    import torch

    return torch.nn.Sequential(
        torch.nn.Linear(dimensao, receita["oculta"]), torch.nn.GELU(),
        torch.nn.Dropout(receita["dropout"]), torch.nn.Linear(receita["oculta"], 1),
    )


def treinar(X_treino: np.ndarray, y_treino: np.ndarray, X_validacao: np.ndarray, y_validacao: np.ndarray,
            paineis_validacao: np.ndarray, *, semente: int, device: str = "cpu",
            receita: dict[str, Any] = RECEITA) -> dict[str, Any]:
    """Treina uma cabeca e devolve o estado da MELHOR avaliacao, a padronizacao e o historico."""
    import torch

    media, desvio = padronizador(X_treino)
    torch.manual_seed(semente)
    dispositivo = torch.device(device)
    xt = torch.tensor((X_treino - media) / desvio, dtype=torch.float32, device=dispositivo)
    yt = torch.tensor(np.asarray(y_treino, dtype=np.float32), device=dispositivo).unsqueeze(1)
    xv = torch.tensor((X_validacao - media) / desvio, dtype=torch.float32, device=dispositivo)
    rede = montar_rede(xt.shape[1], receita).to(dispositivo)
    otimizador = torch.optim.Adam(rede.parameters(), lr=receita["lr"], weight_decay=receita["weight_decay"])
    perda = torch.nn.BCEWithLogitsLoss()
    melhor: dict[str, Any] | None = None
    sem_melhora, historico = 0, []
    for epoca in range(receita["max_epocas"]):
        rede.train()
        otimizador.zero_grad()
        perda(rede(xt), yt).backward()
        otimizador.step()
        if epoca % receita["avaliar_a_cada"]:
            continue
        rede.eval()
        with torch.no_grad():
            sv = rede(xv).squeeze(1).cpu().numpy()
        valor = metricas.macro(metricas.por_painel(sv, y_validacao, paineis_validacao))
        historico.append({"epoca": epoca, "macro_validacao": valor})
        if valor is not None and (melhor is None or valor > melhor["macro_validacao"]):
            melhor = {"epoca": epoca, "macro_validacao": valor,
                      "estado": copy.deepcopy({k: v.detach().cpu() for k, v in rede.state_dict().items()})}
            sem_melhora = 0
        else:
            sem_melhora += 1
            if sem_melhora >= receita["paciencia"]:
                break
    if melhor is None:
        raise CabecaSemSelecao("nenhuma avaliacao teve macro definida na validacao")
    rede.load_state_dict(melhor["estado"])
    rede = rede.cpu().eval()
    return {"rede": rede, "media": media, "desvio": desvio, "epoca": melhor["epoca"],
            "macro_validacao": melhor["macro_validacao"], "historico": historico, "semente": semente}


def pontuar(cabeca: dict[str, Any], X: np.ndarray) -> np.ndarray:
    """Logits da cabeca treinada."""
    import torch

    with torch.no_grad():
        entrada = torch.tensor((X - cabeca["media"]) / cabeca["desvio"], dtype=torch.float32)
        return cabeca["rede"](entrada).squeeze(1).numpy().astype(np.float64)


def _softplus(x: np.ndarray) -> np.ndarray:
    """log(1 + e^x) sem estouro."""
    return np.maximum(x, 0.0) + np.log1p(np.exp(-np.abs(x)))


def _sigmoide(x: np.ndarray) -> np.ndarray:
    """1 / (1 + e^-x) sem estouro nos dois lados."""
    x = np.asarray(x, dtype=np.float64)
    saida = np.empty_like(x)
    positivo = x >= 0
    saida[positivo] = 1.0 / (1.0 + np.exp(-x[positivo]))
    e = np.exp(x[~positivo])
    saida[~positivo] = e / (1.0 + e)
    return saida


def platt(logits: np.ndarray, rotulos: np.ndarray, *, max_iteracoes: int = 100, tolerancia: float = 1e-5,
          passo_minimo: float = 1e-10, sigma: float = 1e-12) -> tuple[float, float]:
    """(a, b) de p = sigmoide(a*s + b): Platt (1999) pelo algoritmo de Lin, Lin e Weng (2007).

    Entropia cruzada com os alvos suavizados de Platt, calculada sem estouro; Newton com busca em linha (Armijo),
    a partir de a = 0 e b = log((n_pos + 1) / (n_neg + 1)). A versao anterior dava passos completos de Newton a
    partir de a = 1: com logits grandes a sigmoide saturava, a curvatura ia a zero e o passo explodia -- em
    [-10, -8, 8, 10] saia a = 6,8e9 e probabilidades 0/1, quando o otimo e a ~ 0,12 (revisao de 23/09).
    """
    s = np.asarray(logits, dtype=np.float64)
    y = np.asarray(rotulos, dtype=int)
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError("Platt precisa das duas classes")
    alvo = np.where(y == 1, (n_pos + 1) / (n_pos + 2), 1 / (n_neg + 2))

    def perda(a: float, b: float) -> float:
        z = a * s + b
        return float(np.sum(alvo * _softplus(-z) + (1 - alvo) * _softplus(z)))

    a, b = 0.0, float(np.log((n_pos + 1) / (n_neg + 1)))
    valor = perda(a, b)
    for _ in range(max_iteracoes):
        p = _sigmoide(a * s + b)
        g_a, g_b = float(np.sum((p - alvo) * s)), float(np.sum(p - alvo))
        if abs(g_a) < tolerancia and abs(g_b) < tolerancia:
            break
        w = p * (1 - p)
        h_aa, h_bb, h_ab = float(np.sum(w * s * s)) + sigma, float(np.sum(w)) + sigma, float(np.sum(w * s))
        determinante = h_aa * h_bb - h_ab * h_ab
        d_a = -(h_bb * g_a - h_ab * g_b) / determinante
        d_b = -(-h_ab * g_a + h_aa * g_b) / determinante
        inclinacao = g_a * d_a + g_b * d_b
        passo = 1.0
        while passo >= passo_minimo:
            novo_a, novo_b = a + passo * d_a, b + passo * d_b
            novo_valor = perda(novo_a, novo_b)
            if novo_valor < valor + 1e-4 * passo * inclinacao:
                a, b, valor = novo_a, novo_b, novo_valor
                break
            passo /= 2
        else:
            break  # nenhuma descida: fica no ultimo ponto aceito, que nunca e pior que o inicial
    return float(a), float(b)


def calibrar(logits: np.ndarray, a: float, b: float) -> np.ndarray:
    return _sigmoide(a * np.asarray(logits, dtype=np.float64) + b)


def limiar_de_mcc(probabilidades: np.ndarray, rotulos: np.ndarray) -> dict[str, float]:
    """O limiar que maximiza o MCC na validacao, para as metricas com limiar do G6/G7 (congelado por sistema)."""
    p = np.asarray(probabilidades, dtype=np.float64)
    y = np.asarray(rotulos, dtype=int)
    melhor = {"limiar": 0.5, "mcc": -1.0}
    for limiar in np.unique(p):
        previsto = p >= limiar
        vp = float(np.sum(previsto & (y == 1)))
        vn = float(np.sum(~previsto & (y == 0)))
        fp = float(np.sum(previsto & (y == 0)))
        fn = float(np.sum(~previsto & (y == 1)))
        denominador = math.sqrt((vp + fp) * (vp + fn) * (vn + fp) * (vn + fn))
        mcc = (vp * vn - fp * fn) / denominador if denominador else 0.0
        if mcc > melhor["mcc"]:
            melhor = {"limiar": float(limiar), "mcc": float(mcc)}
    return melhor


def rodar_sementes(matriz: np.ndarray, tabela: Any, linhas: dict[str, np.ndarray], *, sementes: list[int],
                   device: str = "cpu", receita: dict[str, Any] = RECEITA) -> list[dict[str, Any]]:
    """Uma cabeca por semente: treina no `train`, para e calibra na `validation`, pontua `validation` e `selecao`.

    E a unica funcao que treina cabeca na campanha: o G5 e o comparador M0 x MR passam por aqui, entao o
    procedimento e o mesmo por construcao. Devolve, por semente, as metricas e as probabilidades calibradas.
    """
    y = tabela["binary_label"].astype(int).to_numpy()
    paineis = tabela["primary_panel"].astype(str).to_numpy()
    treino, validacao, selecao = linhas["train"], linhas["validation"], linhas["selecao"]
    for nome, indices in (("train", treino), ("validation", validacao), ("selecao", selecao)):
        if len(np.unique(y[indices])) < 2:
            raise ValueError(f"o papel {nome} precisa das duas classes (tem {len(indices)} variantes)")
    saida = []
    for semente in sementes:
        cabeca = treinar(matriz[treino], y[treino], matriz[validacao], y[validacao], paineis[validacao],
                         semente=int(semente), device=device, receita=receita)
        logits_va, logits_se = pontuar(cabeca, matriz[validacao]), pontuar(cabeca, matriz[selecao])
        a, b = platt(logits_va, y[validacao])
        prob_va, prob_se = calibrar(logits_va, a, b), calibrar(logits_se, a, b)
        saida.append({
            "semente": int(semente), "epoca": cabeca["epoca"], "macro_validacao_na_parada": cabeca["macro_validacao"],
            "platt": {"a": a, "b": b, "inverte_a_ordem": a <= 0},
            "limiar_de_mcc": limiar_de_mcc(prob_va, y[validacao]),
            # Metricas de ORDEM sobre os LOGITS. Platt com a > 0 e monotono e nao muda a ordem, mas sobre a
            # probabilidade a saturacao perto de 0 e 1 viraria empate. As probabilidades calibradas ficam para a
            # media entre sementes e para os limiares.
            "validacao": metricas.resumo(logits_va, y[validacao], paineis[validacao]),
            "selecao": metricas.resumo(logits_se, y[selecao], paineis[selecao]),
            "prob_validacao": prob_va, "prob_selecao": prob_se,
            "modelo": {"estado": {k: v.clone() for k, v in cabeca["rede"].state_dict().items()},
                       "media": cabeca["media"], "desvio": cabeca["desvio"]},
        })
    return saida


def sem_matrizes(rodada: dict[str, Any]) -> dict[str, Any]:
    """A parte de uma rodada que vai para JSON: sem as probabilidades nem o modelo."""
    return {k: v for k, v in rodada.items() if not k.startswith("prob_") and k != "modelo"}


def carregar_cabeca_salva(caminho: Any) -> dict[str, Any]:
    """Uma cabeca gravada pelo comparador, pronta para `pontuar_salva`: rede em `eval()`, com as chaves exatas."""
    import torch

    carga = torch.load(caminho, map_location="cpu", weights_only=False)
    if carga.get("formato") != FORMATO_DA_CABECA:
        raise ValueError(f"{caminho}: formato {carga.get('formato')!r}, esperado {FORMATO_DA_CABECA!r}")
    rede = montar_rede(int(carga["estado"]["0.weight"].shape[1]), carga["receita"])
    rede.load_state_dict(carga["estado"], strict=True)
    return {**carga, "rede": rede.eval()}


def pontuar_salva(cabeca: dict[str, Any], X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(logits, probabilidade calibrada) de uma cabeca recarregada: o MESMO caminho de `rodar_sementes`."""
    logits = pontuar(cabeca, X)
    return logits, calibrar(logits, cabeca["platt"]["a"], cabeca["platt"]["b"])

