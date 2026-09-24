#!/usr/bin/env python3
"""Confere uma corrida de adapter contra a DECLARACAO e imprime a entrada de `adapters_congelados`.

So LE: nada e gravado. A entrada impressa e o que vai para `configs/campanha_r03_desenvolvimento.json`, sem hash
copiado a mao. Nada e aceito porque a corrida disse (`eval/campanha/congelamento.py`, revisao de 24/09):

    - receita e orcamento conferidos no relatorio E nos argumentos gravados no checkpoint, que tem de concordar;
    - passos concluidos, sem retomada, backbone intacto, zero falha de janela, 44.645 janelas de treino e 800 de
      validacao;
    - planos: sha256 registrado no relatorio E recalculado agora dos arquivos, contra o declarado completo; R03
      contra o declarado completo;
    - recorte: o do detalhe da corrida contra o RE-SORTEADO do plano de validacao declarado, com a semente e o limite
      declarados -- nao contra outra corrida;
    - melhora RECALCULADA do historico (menor `focal_alt`, finito, menor que a base), passo e valor conferidos;
    - sha256 de `adapter_melhor.pt` = o do relatorio;
    - se o melhor e o final, os ESTADOS de `adapter.pt` e `adapter_melhor.pt` tem de ser iguais (estado vazio ou
      chaves fora das declaradas nunca contam como iguais). Iguais os estados, os arquivos ainda tem sha256
      diferentes: o script lista os campos que diferem fora do estado, sem atribuir a eles a diferenca inteira.

USO (notebook)
    PYTHONPATH="$WORK" python3 scripts/congelar_adapter.py --corrida ~/artifacts/redesenho/g4_a2 --semente 20260922
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from eval.campanha.cache import sha256_do_arquivo  # noqa: E402
from eval.campanha.congelamento import comparar_estados, conferir_corrida, receita_da_configuracao  # noqa: E402
from eval.campanha.recortes import carregar_campanha  # noqa: E402
from scripts.train_population_adapter import (  # noqa: E402
    amostrar_preservando_a_mistura,
    hash_do_recorte,
    recorte_de_referencia,
)


def campos_diferentes(a: dict[str, Any], b: dict[str, Any]) -> list[str]:
    """Campos de primeiro nivel que diferem entre dois checkpoints, fora do estado do adapter."""
    diferentes = []
    for chave in sorted((set(a) | set(b)) - {"estado_do_adapter"}):
        try:
            if a.get(chave) != b.get(chave):
                diferentes.append(chave)
        except Exception:  # noqa: BLE001 -- comparacao ambigua conta como diferenca
            diferentes.append(chave)
    return diferentes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corrida", required=True, type=Path, help="--out-dir da corrida do adapter")
    parser.add_argument("--semente", required=True, type=int)
    parser.add_argument("--campanha", type=Path, default=Path("configs/campanha_r03_desenvolvimento.json"))
    args = parser.parse_args(argv)

    import pandas as pd
    import torch

    campanha = carregar_campanha(args.campanha)
    declarada = campanha["adapter_do_mr"]["receita"]
    pasta = args.corrida.expanduser()
    relatorio = json.loads((pasta / "treino_do_adapter.json").read_text(encoding="utf-8"))
    melhor_pt = torch.load(pasta / "adapter_melhor.pt", map_location="cpu", weights_only=False)
    config = melhor_pt.get("config") or {}

    caminhos = {papel: Path(str(config.get(f"plano_{papel}", ""))).expanduser() for papel in ("treino", "validacao")}
    ausentes = [str(c) for c in caminhos.values() if not c.is_file()]
    if ausentes:
        print(f"FALHOU: planos gravados no checkpoint nao encontrados: {ausentes}")
        return 2
    amostra = amostrar_preservando_a_mistura(pd.read_parquet(caminhos["validacao"]),
                                             quantos=int(declarada["limite_validacao"]),
                                             seed=int(declarada["seed_da_validacao"]))
    recorte_esperado = hash_do_recorte(zip(amostra["fonte"].astype(str), amostra["variant_id"].astype(str),
                                           amostra["focal_index"].astype(int)))

    entrada, problemas = conferir_corrida(
        relatorio, campanha, semente=args.semente,
        sha256_do_arquivo=sha256_do_arquivo(pasta / "adapter_melhor.pt"),
        recorte=recorte_de_referencia(pasta), recorte_esperado=recorte_esperado,
        sha256_dos_planos={papel: sha256_do_arquivo(caminho) for papel, caminho in caminhos.items()},
        receita_do_checkpoint=receita_da_configuracao(config))

    if entrada is not None:
        final_passo = (relatorio.get("historico") or [{}])[-1].get("passo")
        if entrada["passo"] != final_passo:
            entrada["adapter_pt"] = "o melhor nao e o final: o adapter.pt NAO e o adapter congelado"
        elif not (pasta / "adapter.pt").exists():
            problemas.append("o melhor e o final, mas adapter.pt nao existe para conferir")
        else:
            final_pt = torch.load(pasta / "adapter.pt", map_location="cpu", weights_only=False)
            estados = comparar_estados(melhor_pt.get("estado_do_adapter") or {}, melhor_pt.get("chaves") or [],
                                       final_pt.get("estado_do_adapter") or {}, final_pt.get("chaves") or [],
                                       torch.equal)
            if not estados["iguais"]:
                problemas.append(f"o melhor e o final, mas os estados do adapter diferem ({estados}): investigar "
                                 f"antes de congelar")
            else:
                mesmo_arquivo = sha256_do_arquivo(pasta / "adapter.pt") == entrada["sha256"]
                entrada["adapter_pt"] = {
                    "estados_do_adapter": f"iguais ({estados['chaves']} tensores)",
                    "sha256_dos_arquivos": "iguais" if mesmo_arquivo else "diferentes",
                    "campos_que_diferem_fora_do_estado": campos_diferentes(melhor_pt, final_pt)}

    if problemas:
        print(f"FALHOU: a corrida em {pasta} nao congela pela regra ({len(problemas)} problema(s)):")
        for problema in problemas:
            print(f"  - {problema}")
        return 2
    arquivo = str(pasta / "adapter_melhor.pt").replace(str(Path.home()), "~", 1)
    entrada = {"arquivo": arquivo, **entrada, "congelado_em": date.today().isoformat()}
    print(f"[congelamento] semente {args.semente}: confere com a declaracao (recorte esperado "
          f"{recorte_esperado['sha256'][:16]}, {recorte_esperado['janelas']} janelas)")
    print(json.dumps({str(args.semente): entrada}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
