#!/usr/bin/env python3
"""Fase 0.5 -- cheque da superficie LoRA do apply_lora() sobre o checkpoint R03 REAL.

RODA NO HOST GPU DO R03, DEPOIS de `bash scripts/setup-gpu.sh` + `source scripts/env.sh`
(assim o mamba_ssm.Mamba3 e o kernel ativo e os nomes de submodulo in_proj/out_proj sao os que
o treino vai enxergar; no fallback CPU `mamba3` puro os nomes podem diferir). NAO treina nada.

POR QUE ESTE CHEQUE
-------------------
O apply_lora() deste repo (eval/clinvar/lora.py) e EXCLUSION-based: envolve TODO nn.Linear do
backbone MENOS os que casam _EXCLUDE_PATTERNS (as heads). Cruzei as heads do R03 contra a lista
no Windows e todas estao cobertas -- mas tres coisas so se confirmam com o checkpoint real
instanciado no ambiente de treino:

  1. o Mamba (BidiMambaMidBlock -> mamba_ssm.Mamba3) expoe in_proj/out_proj como nn.Linear que
     entram no LoRA? (e o grosso dos params treinaveis)
  2. a atencao e mista: LocalWindowAttention tem q/k/v/out_proj separados (entram), mas
     SparseGlobalAttention usa nn.MultiheadAttention, que empacota QKV em in_proj_weight
     (um Parameter, NAO um Modulo -> NAO recebe LoRA; so o out_proj recebe). Cobertura assimetrica.
  3. nenhum nn.Linear inesperado do R03 fora das heads e envolvido, e NENHUMA head e envolvida.

Isto decide a superficie treinavel ANTES da Fase 1/2. A paridade "mesma superficie do v10 do
Pedro" NAO e automatica no R03 (arquitetura de atencao diferente) -- este script mostra a real.

USO
---
    python scripts/r03_lora_surface_check.py \\
        --checkpoint s3://croma-bioai-lumina-artifacts-us-east-2/experiments/LUM-20260719-001/runs/R03/checkpoints/final/best_checkpoint.pt \\
        --rank 4 --alpha 8 --dropout 0.1 \\
        --out ~/artifacts/fase05/r03_lora_surface.json
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn

from lumina import load_model_from_checkpoint

from eval.clinvar.lora import LoRALinear, apply_lora

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("r03_lora_surface")

# Heads do R03 (model.py) que o apply_lora NUNCA pode envolver -- a atribuicao limpa depende disso.
R03_HEAD_TOKENS = (
    "mlm_head", "conservation_scalar_head", "conservation_bin_head",
    "splice_class_head", "splice_distance_head", "region_head",
    "counterfactual_snv_head", "missense_severity_head",
    "population_af_head", "population_observed_head",
    "regulatory_head", "hic_left", "hic_right", "cell_film",
)

# Buckets pra categorizar os alvos envolvidos.
BUCKETS = {
    "mamba_in_proj": ("in_proj",),
    "mamba_out_proj": ("out_proj",),  # inclui tambem o out_proj das atencoes; desambiguado abaixo
    "attn_q_proj": ("q_proj",),
    "attn_k_proj": ("k_proj",),
    "attn_v_proj": ("v_proj",),
    "up_gate": (".gate",),
    "stem_purity": ("purity",),
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, help="s3:// ou caminho local do checkpoint R03")
    p.add_argument("--rank", type=int, default=4)
    p.add_argument("--alpha", type=float, default=8.0)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--device", default="cpu", help="cpu basta p/ o cheque estrutural (mas rode no env GPU)")
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args(argv)


def categorize(name: str) -> str:
    for bucket, tokens in BUCKETS.items():
        if any(tok in name for tok in tokens):
            return bucket
    return "other"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    log.info("Carregando R03 de %s (device=%s)", args.checkpoint, args.device)
    model = load_model_from_checkpoint(args.checkpoint, device=torch.device(args.device))

    # Inventario ANTES: todo nn.Linear e todo nn.MultiheadAttention do backbone.
    all_linears = [n for n, m in model.named_modules() if isinstance(m, nn.Linear)]
    mha_modules = [n for n, m in model.named_modules() if isinstance(m, nn.MultiheadAttention)]

    summary = apply_lora(model, rank=args.rank, alpha=args.alpha, dropout=args.dropout)
    wrapped = list(summary.module_names)
    wrapped_set = set(wrapped)

    # Confirma via instancia real quais viraram LoRALinear (defende contra desalinhamento de nome).
    lora_wrapped = [n for n, m in model.named_modules() if isinstance(m, LoRALinear)]

    buckets = Counter(categorize(n) for n in wrapped)

    # Heads que vazaram pro LoRA (tem que ser vazio).
    head_leaks = sorted(n for n in wrapped if any(tok in n for tok in R03_HEAD_TOKENS))

    # MultiheadAttention: o out_proj interno E nn.Linear (subclasse) -> envolvido; o in_proj_weight
    # empacotado e Parameter -> nao. Confirma a assimetria listando o que foi pego por MHA.
    mha_wrapped = sorted(n for n in wrapped if any(n.startswith(f"{mha}.") for mha in mha_modules))

    payload = {
        "checkpoint": str(args.checkpoint),
        "lora_rank": args.rank,
        "n_linears_total": len(all_linears),
        "n_wrapped": len(wrapped),
        "n_lora_linear_instances": len(lora_wrapped),
        "trainable_lora_params": summary.total_params,
        "buckets": dict(buckets),
        "n_multihead_attention_modules": len(mha_modules),
        "mha_wrapped_submodules": mha_wrapped,
        "head_leaks": head_leaks,
        "sample_wrapped": wrapped[:40],
    }

    print(json.dumps(payload, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2))
        log.info("Escrito %s", args.out)

    # Vereditos
    log.info("Linears totais=%d  envolvidos=%d  params LoRA=%d", len(all_linears), len(wrapped), summary.total_params)
    log.info("buckets=%s", dict(buckets))
    if buckets.get("mamba_in_proj", 0) == 0:
        log.warning("NENHUM in_proj envolvido -> o Mamba pode NAO estar recebendo LoRA. Investigar o "
                    "impl do Mamba3 (mamba_ssm vs mamba3 puro) e os nomes de submodulo.")
    else:
        log.info("OK: %d in_proj do Mamba envolvidos.", buckets["mamba_in_proj"])
    if head_leaks:
        log.error("VAZAMENTO: apply_lora envolveu head(s) do R03: %s -> ATUALIZAR _EXCLUDE_PATTERNS.", head_leaks)
        return 1
    log.info("OK: nenhuma head do R03 foi envolvida.")
    if mha_wrapped:
        log.warning("MHA (sparse/anchor attn): %d modulos, %d submodulos envolvidos (%s). ATENCAO: o "
                    "out_proj do nn.MultiheadAttention e envolvido MAS o LoRA provavelmente fica INERTE -- "
                    "o forward do MHA le out_proj.weight/.bias via F.multi_head_attention_forward, sem chamar "
                    "LoRALinear.forward, entao o delta nao entra. Somado ao QKV empacotado (nao envolvido), a "
                    "atencao sparse/anchor recebe ~zero LoRA treinavel. Decidir no port se importa p/ paridade.",
                    len(mha_modules), len(mha_wrapped), mha_wrapped)
    else:
        log.info("MHA: %d modulos, nenhum submodulo envolvido.", len(mha_modules))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
