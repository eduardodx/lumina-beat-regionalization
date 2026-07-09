#!/usr/bin/env python3
"""Phase-1 TRAINING-coupling smoke for Beat-v11 (no data / FASTA / pyfaidx needed).

The real ``train_abraom_frequency_adapter.py`` needs the hg38 FASTA + the ABRAOM
parquet dataset + pyfaidx to *produce* variant sequences. This smoke isolates the
only thing that is v11-specific risk -- does the frequency-adapter training loop
actually couple to the Beat-v11 backbone? -- by feeding SYNTHETIC ref/alt sequences
straight into the same code paths the trainer uses:

    build_finetune_adapter("beat-v11", ...)          # the factory branch (new)
    freeze backbone + apply_lora                      # 105 LoRA linears (Fase-0 verified)
    adapter.extract_variant_features(ref, alt, ...)   # two-tower path (NOT exercised by Fase-0)
    head(cat(site_ref, variant_repr, local_context))  # AbraomFrequencyModel head, d_model*3
    MSE loss -> backward -> AdamW step (bf16 autocast)

Passes iff: forward runs, loss is finite, gradients reach the LoRA + head params,
and the loss actually moves over a few steps. Run on SageMaker (GPU + the r1 ckpt):

    python scripts/smoke_beat_v11_freq_train.py --checkpoint /tmp/r1_best.pt
"""

from __future__ import annotations

# --- tilelang shim: before ANY import that pulls mamba_ssm ---
import sys as _sys
import types as _types

if not isinstance(_sys.modules.get("tilelang"), _types.ModuleType):
    _sys.modules["tilelang"] = None

from pathlib import Path as _Path

_REPO_ROOT = _Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))
# -------------------------------------------------------------

import argparse
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_CKPT = "s3://ai4bio-lumina/releases/lumina-beat-v11v5-r1-202607071631/ckpt/best_checkpoint.pt"


def _rand_dna(n: int) -> str:
    return "".join(random.choice("ACGT") for _ in range(n))


def _check(label: str, ok: bool, detail: str = "") -> bool:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}{(' -- ' + detail) if detail else ''}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--context-size", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--lora-rank", type=int, default=8)
    args = ap.parse_args()
    device = torch.device(args.device)

    from eval.clinvar.adapters import build_finetune_adapter
    from eval.clinvar.lora import apply_lora

    ok = True

    # ---- factory path (mirrors build_model in the trainer) ----------------------------------
    print(f"[build] build_finetune_adapter('beat-v11', ...) from {args.checkpoint}")
    adapter = build_finetune_adapter(
        "beat-v11", "r1", device, checkpoint_path=args.checkpoint,
    )
    d = adapter.d_model
    ok &= _check("factory returned a beat-v11 adapter (d_model)", d > 0, f"d_model={d}")

    for p in adapter.backbone.parameters():
        p.requires_grad_(False)
    summary = apply_lora(adapter.backbone, rank=args.lora_rank, alpha=16.0, dropout=0.05)
    print(f"    apply_lora: {summary.module_count} linears, {summary.total_params:,} params")

    # ---- head, identical shape to AbraomFrequencyModel (d_model*3 -> 1) ----------------------
    head = nn.Sequential(
        nn.LayerNorm(d * 3), nn.Linear(d * 3, 256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256, 1)
    ).to(device)

    lora_params = [p for n, p in adapter.backbone.named_parameters() if "lora_" in n and p.requires_grad]
    params = lora_params + list(head.parameters())
    ok &= _check("trainable LoRA params present", len(lora_params) > 0, f"{len(lora_params)} tensors")
    opt = torch.optim.AdamW(params, lr=1e-3)

    # ---- synthetic batch: ref/alt differ by one base at the center ---------------------------
    L, B = args.context_size, args.batch
    off = L // 2
    ref_seqs = [_rand_dna(L) for _ in range(B)]
    alt_seqs, ref_alleles, alt_alleles = [], [], []
    for s in ref_seqs:
        alt_base = "T" if s[off] != "T" else "A"
        alt_seqs.append(s[:off] + alt_base + s[off + 1:])
        ref_alleles.append(s[off])
        alt_alleles.append(alt_base)
    variant_offsets = [off] * B
    target = torch.rand(B, device=device)  # stand-in for logit(af_abraom)

    # ---- train loop -------------------------------------------------------------------------
    print(f"[train] {args.steps} steps, B={B}, L={L}, bf16 autocast")
    losses: list[float] = []
    head.train()
    adapter.backbone.train()
    for step in range(args.steps):
        opt.zero_grad()
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            site_ref, variant_repr, local_context = adapter.extract_variant_features(
                ref_seqs, alt_seqs, variant_offsets, ref_alleles, alt_alleles
            )
            feat = torch.cat([site_ref, variant_repr, local_context], dim=-1)
            ok_shape = feat.shape[-1] == d * 3
            pred = head(feat).squeeze(-1)
            loss = F.mse_loss(pred.float(), target)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(params, 1e9)  # measure, don't clip
        opt.step()
        losses.append(float(loss.item()))
        print(f"    step {step}: loss={loss.item():.5f}  grad_norm={float(gnorm):.4f}")

    ok &= _check("feature width == d_model*3", ok_shape, f"{feat.shape[-1]} vs {d*3}")
    ok &= _check("all losses finite", all(math.isfinite(x) for x in losses), str(losses))
    lora_grad = any(p.grad is not None and torch.isfinite(p.grad).all() for p in lora_params)
    head_grad = all(p.grad is not None for p in head.parameters())
    ok &= _check("gradients reach LoRA params", lora_grad)
    ok &= _check("gradients reach head params", head_grad)
    ok &= _check("loss moved over steps", abs(losses[0] - losses[-1]) > 1e-6, f"{losses[0]:.5f} -> {losses[-1]:.5f}")

    print("\nRESULT:", "ALL PASS" if ok else "FAILURES ABOVE")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
