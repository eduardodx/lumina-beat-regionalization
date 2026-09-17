"""Fidelidade do extrator portado: o codigo executavel tem de ser IGUAL ao do commit de origem.

Os testes de comportamento (`test_embedding_probe_rich_port.py`) comparam algumas funcoes utilitarias e precisam
de torch. Este aqui cobre o arquivo INTEIRO -- `head_readouts`, `MidStackTaps`, `assert_r03_head_layout` e os hooks
inclusive -- comparando o texto, e so precisa de git e da stdlib: roda no Windows e no notebook.

A unica diferenca aceita e o cabecalho de proveniencia inserido no port.

    PYTHONPATH=. python3 tests/test_rich_port_fidelity.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
PORTADO = RAIZ / "eval" / "embedding_probe" / "rich.py"
ORIGEM_COMMIT = "23fb518997ee7c68d843ae55bc2e1409ddf24cde"
ORIGEM_CAMINHO = "eval/embedding_probe/rich.py"
MARCA_FIM_DO_CABECALHO = "Docstring original abaixo.\n\n"


class Skip(Exception):
    """Teste nao executado por falta de dependencia. NAO e um PASS -- o runner conta separado."""


def _origem() -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(RAIZ), "show", f"{ORIGEM_COMMIT}:{ORIGEM_CAMINHO}"],
            check=True, capture_output=True, text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise Skip(f"commit de origem {ORIGEM_COMMIT[:12]} indisponivel neste clone: {exc}") from exc


def _sem_cabecalho(portado: str) -> str:
    """Devolve o portado com o cabecalho de proveniencia trocado pela abertura original da docstring."""
    assert MARCA_FIM_DO_CABECALHO in portado, "cabecalho de proveniencia ausente ou alterado"
    corpo = portado.split(MARCA_FIM_DO_CABECALHO, 1)[1]
    return '"""' + corpo


def test_cabecalho_de_proveniencia_aponta_o_commit_certo():
    texto = PORTADO.read_text(encoding="utf-8")
    assert ORIGEM_COMMIT in texto, "o cabecalho tem de citar o commit de origem"
    assert "SEM alteracao de comportamento" in texto


def test_codigo_executavel_identico_ao_da_origem():
    portado = PORTADO.read_text(encoding="utf-8").replace("\r\n", "\n")
    origem = _origem().replace("\r\n", "\n")
    reconstruido = _sem_cabecalho(portado)
    if reconstruido != origem:
        linhas_p = reconstruido.splitlines()
        linhas_o = origem.splitlines()
        diferentes = [i for i, (a, b) in enumerate(zip(linhas_p, linhas_o)) if a != b]
        detalhe = (f"{len(diferentes)} linha(s) diferentes, primeira na {diferentes[0] + 1}: "
                   f"{linhas_p[diferentes[0]]!r} != {linhas_o[diferentes[0]]!r}") if diferentes else \
                  f"tamanhos diferentes: {len(linhas_p)} x {len(linhas_o)} linhas"
        raise AssertionError(f"o portado divergiu da origem: {detalhe}")


def test_o_arquivo_traz_as_pecas_que_o_smoke_vai_usar():
    texto = PORTADO.read_text(encoding="utf-8")
    for peca in ("def head_readouts", "class MidStackTaps", "def assert_r03_head_layout",
                 "def substitution_onehot", "def pooled"):
        assert peca in texto, peca


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed, skipped = 0, []
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Skip as exc:
            skipped.append(name)
            print(f"  SKIP  {name}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
    ran = len(tests) - failed - len(skipped)
    tail = f"  |  {len(skipped)} PULADO(S), sem cobertura: {', '.join(skipped)}" if skipped else ""
    print(f"\n{ran}/{len(tests) - len(skipped)} passaram{tail}")
    if skipped and os.environ.get("REQUIRE_NO_SKIP"):
        print("REQUIRE_NO_SKIP: teste pulado conta como falha")
        sys.exit(1)
    sys.exit(1 if failed else 0)
