"""Leituras do G3 sobre um modelo de brinquedo com o layout de cabecas do R03. Precisa de torch.

    PYTHONPATH=. REQUIRE_NO_SKIP=1 python3 tests/test_campanha_leituras.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class Skip(Exception):
    """Teste nao executado por falta de dependencia. NAO e um PASS."""


try:
    import torch
    from torch import nn

    from eval.campanha.leituras import ler_extracoes
    from eval.embedding_probe.rich import LINEAR_HEADS, R03_HEAD_LAYOUT, substitution_onehot
    TORCH = True
except ImportError as exc:  # pragma: no cover - no Windows
    TORCH = False
    MOTIVO = str(exc)

D = 448


def _exige_torch():
    if not TORCH:
        raise Skip(f"sem torch: {MOTIVO}")


def _modelo():
    """As 7 cabecas lineares com as larguras do R03 e as 3 MLP (5 + 1 + 4 = 10), como em model.py."""
    torch.manual_seed(0)
    modelo = nn.Module()
    for nome, largura in R03_HEAD_LAYOUT:
        setattr(modelo, nome, nn.Linear(D, largura))
    modelo.splice_class_head = nn.Sequential(nn.Linear(D, 64), nn.GELU(), nn.Linear(64, 5))
    modelo.splice_distance_head = nn.Sequential(nn.Linear(D, 64), nn.GELU(), nn.Linear(64, 1))
    modelo.missense_severity_head = nn.Linear(D, 4)
    return modelo


def _post(n_pares: int, comprimento: int = 4096):
    torch.manual_seed(1)
    return torch.randn(2 * n_pares, comprimento, D)


def test_dimensoes_e_so_as_variantes_reais():
    _exige_torch()
    post = _post(4)
    saida = ler_extracoes(_modelo(), post, focal=2047, reais=3, refs=["A", "C", "G"], alts=["G", "T", "A"])
    assert saida["cabecas_172"].shape == (3, 172), saida["cabecas_172"].shape
    assert saida["leitura_antiga_1344"].shape == (3, 1344)


def test_cabecas_lineares_sao_w_delta_e_o_one_hot_fecha_o_bloco():
    _exige_torch()
    modelo, post = _modelo(), _post(2)
    saida = ler_extracoes(modelo, post, focal=2047, reais=2, refs=["A", "C"], alts=["G", "T"])
    delta = post[1, 2047] - post[0, 2047]
    esperado = torch.cat([delta @ getattr(modelo, nome).weight.T for nome in LINEAR_HEADS])
    assert torch.allclose(saida["cabecas_172"][0, :68], esperado, atol=1e-5)
    assert saida["cabecas_172"][0, 156:].tolist() == substitution_onehot("A", "G")


def test_leitura_antiga_e_site_delta_e_media_local():
    _exige_torch()
    post = _post(2)
    saida = ler_extracoes(_modelo(), post, focal=2047, reais=2, refs=["A", "C"], alts=["G", "T"])
    antiga = saida["leitura_antiga_1344"][1]
    ref, alt = post[2], post[3]
    assert torch.allclose(antiga[:D], ref[2047])
    assert torch.allclose(antiga[D:2 * D], alt[2047] - ref[2047])
    assert torch.allclose(antiga[2 * D:], ref[1983:2111].mean(dim=0), atol=1e-6), "media em [f-64, f+64)"


def test_lote_impar_e_recusado():
    _exige_torch()
    try:
        ler_extracoes(_modelo(), torch.randn(3, 16, D), focal=5, reais=1, refs=["A"], alts=["G"])
    except ValueError:
        return
    raise AssertionError("post com numero impar de linhas deveria ser recusado")


if __name__ == "__main__":
    testes = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    falhas, pulados = 0, []
    for nome, funcao in testes:
        try:
            funcao()
            print(f"  PASS  {nome}")
        except Skip as exc:
            pulados.append(nome)
            print(f"  SKIP  {nome}: {exc}")
        except Exception as exc:  # noqa: BLE001
            falhas += 1
            print(f"  FAIL  {nome}: {type(exc).__name__}: {exc}")
    print(f"\n{len(testes) - falhas - len(pulados)}/{len(testes) - len(pulados)} passaram"
          + (f"  |  {len(pulados)} PULADO(S)" if pulados else ""))
    if pulados and os.environ.get("REQUIRE_NO_SKIP"):
        print("REQUIRE_NO_SKIP: teste pulado conta como falha")
        sys.exit(1)
    sys.exit(1 if falhas else 0)
