#!/usr/bin/env python3
"""Confere uma corrida de adapter pela regra declarada e imprime a entrada de `adapters_congelados`.

So LE: nada e gravado. A entrada impressa e o que vai para `configs/campanha_r03_desenvolvimento.json`, sem copiar
hash a mao.

Confere (`eval/campanha/congelamento.py`):
    - a semente e uma das declaradas e a receita e o orcamento sao os declarados (do relatorio; o que ele nao
      registrou vem dos argumentos gravados no checkpoint -- caso da a_1);
    - passos concluidos, backbone intacto, zero falha de janela;
    - os planos declarados, o R03 do contrato e o recorte de 800 janelas (calculado do detalhe da validacao);
    - `adapter_melhor.pt` supera a base e o sha256 do arquivo e o do relatorio.
E descreve, sem decidir nada: se o melhor e o final, compara os TENSORES de `adapter.pt` e `adapter_melhor.pt` --
o sha256 dos dois difere tambem por metadado (data de criacao, identidades), e isso fecha a nota da a_1.

USO (notebook)
    PYTHONPATH="$WORK" python3 scripts/congelar_adapter.py --corrida ~/artifacts/redesenho/g4_a2 --semente 20260922
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from eval.campanha.cache import sha256_do_arquivo  # noqa: E402
from eval.campanha.congelamento import conferir_corrida, receita_da_configuracao  # noqa: E402
from eval.campanha.recortes import carregar_campanha  # noqa: E402
from scripts.extract_campaign_features import PREFIXO_DO_CHECKPOINT  # noqa: E402
from scripts.train_population_adapter import recorte_de_referencia  # noqa: E402


def comparar_tensores(melhor: dict, final: dict) -> dict:
    import torch

    a, b = melhor.get("estado_do_adapter") or {}, final.get("estado_do_adapter") or {}
    if set(a) != set(b):
        return {"identicos": False, "motivo": "conjuntos de chaves diferentes"}
    diferentes = sorted(k for k in a if not torch.equal(a[k], b[k]))
    return {"identicos": not diferentes, "chaves": len(a), "diferentes": diferentes[:5]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corrida", required=True, type=Path, help="--out-dir da corrida do adapter")
    parser.add_argument("--semente", required=True, type=int)
    parser.add_argument("--campanha", type=Path, default=Path("configs/campanha_r03_desenvolvimento.json"))
    args = parser.parse_args(argv)

    import torch

    pasta = args.corrida.expanduser()
    relatorio = json.loads((pasta / "treino_do_adapter.json").read_text(encoding="utf-8"))
    melhor_pt = torch.load(pasta / "adapter_melhor.pt", map_location="cpu", weights_only=False)
    entrada, problemas = conferir_corrida(
        relatorio, carregar_campanha(args.campanha), semente=args.semente,
        sha256_do_arquivo=sha256_do_arquivo(pasta / "adapter_melhor.pt"),
        recorte=recorte_de_referencia(pasta), prefixo_do_checkpoint=PREFIXO_DO_CHECKPOINT,
        receita_do_checkpoint=receita_da_configuracao(melhor_pt.get("config") or {}))
    if problemas:
        print(f"FALHOU: a corrida em {pasta} nao congela pela regra ({len(problemas)} problema(s)):")
        for problema in problemas:
            print(f"  - {problema}")
        return 2

    final_passo = (relatorio.get("historico") or [{}])[-1].get("passo")
    if entrada["passo"] == final_passo and (pasta / "adapter.pt").exists():
        tensores = comparar_tensores(melhor_pt, torch.load(pasta / "adapter.pt", map_location="cpu",
                                                            weights_only=False))
        entrada["adapter_pt"] = ("o melhor e o final: tensores IDENTICOS aos do adapter.pt; o sha256 difere so por "
                                 "metadado" if tensores["identicos"] else f"tensores DIFERENTES: {tensores}")
    else:
        entrada["adapter_pt"] = "o melhor nao e o final: adapter.pt NAO e o adapter congelado"
    arquivo = str(pasta / "adapter_melhor.pt").replace(str(Path.home()), "~", 1)
    entrada = {"arquivo": arquivo, **entrada, "congelado_em": date.today().isoformat()}
    print(f"[congelamento] semente {args.semente}: confere com a regra e com a receita declarada")
    print(json.dumps({str(args.semente): entrada}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
