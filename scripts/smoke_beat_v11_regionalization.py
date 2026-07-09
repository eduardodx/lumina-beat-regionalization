#!/usr/bin/env python3
"""Phase-0 smoke check for the Beat-v11 ABRAOM regionalization port.

Validates the three things Phase-0 needs BEFORE any adapter/M0 training:

  1. forward parity   -> last_hidden_state[..., 320], mid_hidden_state[..., 256],
                         mlm_logits[..., 4], gnomad_af_pred[..., 4]
  2. adapter path     -> FineTuneBeatV11Adapter.forward_hidden_states -> [..., 320]
  3. LoRA coupling    -> apply_lora wraps the mamba/attn linears and NOT the
                         population/conservation/mlm/... heads NOR the MoE (router +
                         experts, frozen for v10 parity), then the model still forwards.

Run on SageMaker (needs GPU for the Mamba3 MIMO kernel + S3 for the checkpoint;
the Windows dev box has neither):

    python scripts/smoke_beat_v11_regionalization.py \
        --checkpoint s3://ai4bio-lumina/releases/lumina-beat-v11v5-r1-202607071631/ckpt/best_checkpoint.pt

Exits non-zero on the first failed invariant.
"""

from __future__ import annotations

# --- tilelang shim: MUST run before ANY import that pulls mamba_ssm (see beat_v11_adapter) ---
import sys as _sys
import types as _types

if not isinstance(_sys.modules.get("tilelang"), _types.ModuleType):
    _sys.modules["tilelang"] = None
# --------------------------------------------------------------------------------------------

import argparse
import logging
import random

import torch

DEFAULT_CKPT = (
    "s3://ai4bio-lumina/releases/lumina-beat-v11v5-r1-202607071631/ckpt/best_checkpoint.pt"
)

# Module-name substrings that apply_lora MUST NOT wrap on Beat-v11: prediction heads
# (v11 renamed several vs v10) plus the MoE router (routing stability). Experts are
# intentionally left LoRA-able (extra adapted capacity).
_FORBIDDEN_LORA_MARKERS = (
    "population_af_head", "population_observed_head",
    "conservation_scalar_head", "conservation_bin_head", "missense_severity_head",
    "mlm_head", "region_head", "splice_class_head", "splice_distance_head",
    "counterfactual_snv_head", "regulatory_head", "hic_left", "hic_right", "cell_film",
    ".router", ".experts.",  # whole MoE frozen for the v10-parity first run
)


def _random_dna(n: int) -> str:
    return "".join(random.choice("ACGT") for _ in range(n))


def _check(label: str, ok: bool, detail: str = "") -> bool:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}{(' -- ' + detail) if detail else ''}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--lora-rank", type=int, default=8)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    device = torch.device(args.device)
    if device.type != "cuda":
        print("WARNING: Mamba3 MIMO kernel needs CUDA; a CPU run will likely fail at forward.")

    from lumina_beat_v11 import SNV_BASES
    from eval.clinvar.beat_v11_adapter import FineTuneBeatV11Adapter
    from eval.clinvar.lora import apply_lora

    ok = True

    # ---- load via the package loader (the adapter does this) --------------------------------
    print(f"[load] {args.checkpoint} on {device}")
    adapter = FineTuneBeatV11Adapter(args.checkpoint, device)
    model = adapter.backbone
    model.eval()
    ok &= _check("d_full == 320", adapter.d_model == 320, f"got {adapter.d_model}")

    seq = _random_dna(args.seq_len)
    batch = adapter.tokenize([seq])
    input_ids = batch["input_ids"]

    # ---- 1) raw forward parity --------------------------------------------------------------
    print("[1/3] forward parity")
    with torch.no_grad():
        out = model(input_ids=input_ids)
    lhs = out["last_hidden_state"]
    mid = out["mid_hidden_state"]
    mlm = out["mlm_logits"]
    afp = out["gnomad_af_pred"]
    ok &= _check("last_hidden_state[..., 320]", lhs.shape[-1] == 320, str(tuple(lhs.shape)))
    ok &= _check("mid_hidden_state[..., 256]", mid.shape[-1] == 256, str(tuple(mid.shape)))
    ok &= _check(f"mlm_logits[..., {len(SNV_BASES)}]", mlm.shape[-1] == len(SNV_BASES), str(tuple(mlm.shape)))
    ok &= _check("gnomad_af_pred[..., 4]", afp.shape[-1] == len(SNV_BASES), str(tuple(afp.shape)))
    ok &= _check("mid length == L/4", mid.shape[1] == lhs.shape[1] // 4, f"{mid.shape[1]} vs {lhs.shape[1]}//4")

    # ---- 2) adapter hidden-state path -------------------------------------------------------
    print("[2/3] adapter forward_hidden_states")
    with torch.no_grad():
        hidden = adapter.forward_hidden_states(batch)
    ok &= _check("forward_hidden_states -> [..., 320]", hidden.shape[-1] == 320, str(tuple(hidden.shape)))

    # ---- 3) LoRA coupling -------------------------------------------------------------------
    print("[3/3] apply_lora coupling")
    # Enumerate the linear surface BEFORE wrapping (avoids counting each LoRALinear's .base).
    all_linears = [n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)]
    moe_linears = [n for n in all_linears if ".experts." in n or ".router" in n]
    print(f"    linear surface: {len(all_linears)} nn.Linear total; "
          f"MoE = {len(moe_linears)} (router+experts), frozen under the v10-parity policy")

    summary = apply_lora(model, rank=args.lora_rank)
    names = list(summary.module_names)
    wrapped_forbidden = sorted({m for n in names for m in _FORBIDDEN_LORA_MARKERS if m in n})
    ok &= _check(
        "LoRA wraps neither heads nor MoE (router+experts) -- v10-parity surface",
        not wrapped_forbidden,
        f"leaked: {wrapped_forbidden}" if wrapped_forbidden else f"{len(names)} modules wrapped",
    )
    groups: dict[str, int] = {}
    for n in names:
        key = n.split(".")[-1]
        groups[key] = groups.get(key, 0) + 1
    print(f"    wrapped {len(names)} linears, {summary.total_params:,} trainable LoRA params")
    for key, count in sorted(groups.items(), key=lambda kv: -kv[1]):
        print(f"      {count:4d}  *.{key}")
    print(f"    NOTE: MoE frozen for v10 parity; the 'LoRA-the-experts' ablation would add the "
          f"{len(moe_linears)} MoE linears above (drop '.experts.' from lora._EXCLUDE_PATTERNS).")

    # wrapped model must still forward
    with torch.no_grad():
        out2 = model(input_ids=input_ids)
    ok &= _check("LoRA'd model still forwards -> [..., 320]",
                 out2["last_hidden_state"].shape[-1] == 320,
                 str(tuple(out2["last_hidden_state"].shape)))

    print("\nRESULT:", "ALL PASS" if ok else "FAILURES ABOVE")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
