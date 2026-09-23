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


class CabecaSemSelecao(RuntimeError):
    """Nenhuma avaliacao teve a macro definida na validacao: nao ha epoca a escolher."""


def padronizador(treino: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    media = treino.mean(axis=0)
    desvio = treino.std(axis=0)
    desvio[desvio < 1e-8] = 1.0  # dimensao constante no treino: nao escala, nao explode
    return media, desvio


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
    rede = torch.nn.Sequential(
        torch.nn.Linear(xt.shape[1], receita["oculta"]), torch.nn.GELU(),
        torch.nn.Dropout(receita["dropout"]), torch.nn.Linear(receita["oculta"], 1),
    ).to(dispositivo)
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


def platt(logits: np.ndarray, rotulos: np.ndarray, *, iteracoes: int = 100) -> tuple[float, float]:
    """(a, b) de p = sigmoide(a*s + b) por Newton na verossimilhanca, com os alvos suavizados de Platt."""
    s = np.asarray(logits, dtype=np.float64)
    y = np.asarray(rotulos, dtype=np.float64)
    n_pos, n_neg = y.sum(), len(y) - y.sum()
    alvo = np.where(y == 1, (n_pos + 1) / (n_pos + 2), 1 / (n_neg + 2))
    a, b = 1.0, 0.0
    for _ in range(iteracoes):
        p = 1 / (1 + np.exp(-(a * s + b)))
        gradiente = np.array([np.sum((p - alvo) * s), np.sum(p - alvo)])
        w = p * (1 - p) + 1e-12
        hessiana = np.array([[np.sum(w * s * s), np.sum(w * s)], [np.sum(w * s), np.sum(w)]])
        passo = np.linalg.solve(hessiana + 1e-9 * np.eye(2), gradiente)
        a, b = a - passo[0], b - passo[1]
        if np.max(np.abs(passo)) < 1e-10:
            break
    return float(a), float(b)


def calibrar(logits: np.ndarray, a: float, b: float) -> np.ndarray:
    return 1 / (1 + np.exp(-(a * np.asarray(logits, dtype=np.float64) + b)))


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
            "platt": {"a": a, "b": b}, "limiar_de_mcc": limiar_de_mcc(prob_va, y[validacao]),
            "validacao": metricas.resumo(prob_va, y[validacao], paineis[validacao]),
            "selecao": metricas.resumo(prob_se, y[selecao], paineis[selecao]),
            "prob_validacao": prob_va, "prob_selecao": prob_se,
        })
    return saida

