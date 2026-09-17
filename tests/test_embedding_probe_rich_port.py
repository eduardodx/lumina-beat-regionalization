"""Equivalencia do extrator portado com o codigo da pesquisa.

O extrator de `eval/embedding_probe/rich.py` foi copiado da branch `embedding-probe-mosaic` sem alteracao de
comportamento. Este teste carrega o arquivo DAQUELE commit via `git show`, importa os dois lado a lado e compara
as saidas nas mesmas entradas. Se alguem mexer no portado sem querer, aqui quebra.

Cobertura: estas comparacoes sao das funcoes utilitarias. O arquivo INTEIRO (`head_readouts`, `MidStackTaps`,
`assert_r03_head_layout`, os hooks) e coberto por `test_rich_port_fidelity.py`, que compara o texto e nao precisa
de torch. A integracao com o R03 de verdade so fica provada no smoke do M0.

Precisa de torch e de um clone com a branch de pesquisa: roda no notebook
(`PYTHONPATH=. pytest tests/test_embedding_probe_rich_port.py -q`), nao no Windows.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import torch

import eval.embedding_probe.rich as portado

ORIGEM_COMMIT = "23fb518997ee7c68d843ae55bc2e1409ddf24cde"
ORIGEM_CAMINHO = "eval/embedding_probe/rich.py"
RAIZ = Path(__file__).resolve().parents[1]


def _carrega_origem():
    """Importa o rich.py do commit de origem, sem trocar de branch nem tocar na arvore de trabalho."""
    try:
        fonte = subprocess.run(
            ["git", "-C", str(RAIZ), "show", f"{ORIGEM_COMMIT}:{ORIGEM_CAMINHO}"],
            check=True, capture_output=True, text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        # FALHA, nao skip: o commit faz parte da historia deste repositorio. Pular deixaria `pytest -q` verde
        # sem que a equivalencia tivesse sido verificada.
        pytest.fail(f"commit de origem {ORIGEM_COMMIT[:12]} indisponivel neste clone: {exc}")

    with tempfile.TemporaryDirectory() as tmp:
        caminho = Path(tmp) / "rich_origem.py"
        caminho.write_text(fonte, encoding="utf-8")
        spec = importlib.util.spec_from_file_location("rich_origem", caminho)
        modulo = importlib.util.module_from_spec(spec)
        sys.modules["rich_origem"] = modulo
        spec.loader.exec_module(modulo)
        return modulo


@pytest.fixture(scope="module")
def origem():
    return _carrega_origem()


def test_constantes_identicas(origem):
    for nome in ("LINEAR_HEADS", "MLP_HEADS", "BASES", "COMPLEMENT", "SUBSTITUTIONS"):
        assert getattr(portado, nome) == getattr(origem, nome), nome


def test_substitution_onehot_identico(origem):
    for ref in portado.BASES:
        for alt in portado.BASES:
            if ref == alt:
                continue
            assert portado.substitution_onehot(ref, alt) == origem.substitution_onehot(ref, alt), (ref, alt)


def test_reverse_complement_identico(origem):
    for seq in ("ACGT", "AACCGGTT", "N", "", "acgt"):
        assert portado.reverse_complement(seq) == origem.reverse_complement(seq), seq


def test_mid_index_identico(origem):
    for focal in (0, 1, 3, 4, 2047, 2048, 4095):
        for downsample in (1, 2, 4):
            assert portado.mid_index(focal, downsample) == origem.mid_index(focal, downsample), (focal, downsample)


def test_pooled_identico(origem):
    torch.manual_seed(0)
    delta = torch.randn(3, 512, 8)
    for raio in (0, 1, 64, 300):
        for reduce in ("mean", "max"):
            a = portado.pooled(delta, focal=256, radius=raio, reduce=reduce)
            b = origem.pooled(delta, focal=256, radius=raio, reduce=reduce)
            assert torch.equal(a, b), (raio, reduce)


def test_pooled_nas_bordas_identico(origem):
    delta = torch.arange(2 * 16 * 2, dtype=torch.float32).reshape(2, 16, 2)
    for focal in (0, 15):
        a = portado.pooled(delta, focal=focal, radius=8)
        b = origem.pooled(delta, focal=focal, radius=8)
        assert torch.equal(a, b), focal
