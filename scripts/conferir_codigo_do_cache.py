#!/usr/bin/env python3
"""Antes de uma extracao longa: o codigo e o ambiente que vao rodar sao os MESMOS que produziram um cache de
referencia?

O MR de cada semente de adapter so se compara ao M0 se as identidades diferirem APENAS pelo adapter
(`conferir_par`). Um arquivo de `ARQUIVOS_QUE_DETERMINAM_AS_FEATURES` mudado, outro pacote `lumina` ou outra
versao de torch/CUDA so apareceriam no comparador, depois de ~3,5 h de extracao. Aqui a conferencia leva segundos.

Nao extrai nada e nao grava nada.

USO (notebook, na GPU que vai extrair)
    PYTHONPATH="$WORK" python3 scripts/conferir_codigo_do_cache.py --cache ~/artifacts/redesenho/g3_cache/M0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))


def diferencas(referencia: dict[str, Any], atual: dict[str, Any], prefixo: str = "") -> list[str]:
    """Chaves (achatadas) cujo valor difere entre a identidade de referencia e a atual. Pura."""
    saida = []
    for chave in sorted(set(referencia) | set(atual)):
        nome = f"{prefixo}{chave}"
        a, b = referencia.get(chave), atual.get(chave)
        if isinstance(a, dict) and isinstance(b, dict):
            saida += diferencas(a, b, f"{nome}.")
        elif a != b:
            saida.append(f"{nome}: {a!r} -> {b!r}")
    return saida


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache", required=True, type=Path, help="cache de referencia (ex.: o do M0)")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    import torch

    from scripts.extract_campaign_features import ambiente_de_execucao, codigo_da_extracao

    identidade = json.loads((args.cache.expanduser() / "identidade.json").read_text(encoding="utf-8"))
    problemas = (diferencas(identidade.get("codigo") or {}, codigo_da_extracao(), "codigo.")
                 + diferencas(identidade.get("ambiente") or {}, ambiente_de_execucao(torch.device(args.device)),
                              "ambiente."))
    arquivos = (identidade.get("codigo") or {}).get("arquivos") or {}
    if problemas:
        print(f"FALHOU: o codigo ou o ambiente atual difere do que produziu {args.cache}:")
        for problema in problemas:
            print(f"  - {problema}")
        print("Uma extracao agora NAO pareia com esse cache. Volte o arquivo (git) ou o ambiente antes de extrair.")
        return 2
    print(f"OK: os {len(arquivos)} arquivos que determinam as features, o pacote lumina e o ambiente sao os "
          f"mesmos de {args.cache}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
