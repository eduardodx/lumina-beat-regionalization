#!/usr/bin/env python3
"""Gera o contrato experimental pre-registrado (§2 do Eduardo) — item (c), fecha a Fase 1.

O §2 exige um manifesto com as identidades cientificas (hashes) CONGELADAS antes dos runs de M1-M4:
foundation checkpoint, genoma de referencia, ClinVar/ABraOM/gnomAD, e os splits. Este script COLETA
os hashes de forma reproduzivel (nao a mao) e emite o manifesto. O que ja existe (Fase 0/1) e
preenchido; o que depende da Fase 2 (gnomAD, ABraOM final) fica marcado PENDING.

Fontes dos hashes:
  * splits + test sets (T_BR, T_nonBR) -> REUSA o splits_manifest.json da Fase 0 (sha256 das
    variant_keys via sha256_keys; ja congelado, nao recalculamos).
  * adapter ClinVar C (best_model.pt) + calibration_m0.json -> sha256 do arquivo (identidade do C
    reusado em M0-M4, §4.2). Registra tambem a RECEITA do C (config do bundle), que §4.2/§4.4 exigem
    identica em M1-M4.
  * foundation checkpoint R03 -> file_sha256 + model_config_sha256 + resolved_config_sha256
    (extraidos via lumina.checkpoint; o Eduardo distingue os dois no §2).
  * fasta hg38 -> sha256 do arquivo (a menos de --skip-fasta-hash; ~3GB, ~1min).

Rodar (no notebook):
    cd "$WORK" && PYTHONPATH="$WORK" "$PY" scripts/build_contract_manifest.py \
        --splits-manifest ~/artifacts/fase0/clinvar_splits/splits_manifest.json \
        --clinvar-adapter ~/artifacts/fase1/m0_full/best_model.pt \
        --calibration ~/artifacts/fase1/m0_full/calibration_m0.json \
        --checkpoint "$R03_CKPT" --fasta ~/hg38/hg38.fa \
        --out ~/artifacts/fase1/m0_full/contract_manifest.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PENDING = "PENDING"


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def sha256_json(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def checkpoint_config_hashes(checkpoint_path: str) -> dict[str, str]:
    """model_config_sha256 (dict cru do checkpoint) + resolved_config_sha256 (LuminaConfig resolvido)."""
    try:
        from lumina.checkpoint import config_from_checkpoint, torch_load_checkpoint

        ckpt = torch_load_checkpoint(checkpoint_path, map_location="cpu")
        raw = ckpt.get("config", {})
        model_config = raw.get("model_config") if isinstance(raw, dict) else None
        if not isinstance(model_config, dict):
            model_config = ckpt.get("model_config") if isinstance(ckpt.get("model_config"), dict) else {}
        resolved = config_from_checkpoint(ckpt)
        return {
            "model_config_sha256": sha256_json(model_config) if model_config else PENDING,
            "resolved_config_sha256": sha256_json(asdict(resolved)),
        }
    except Exception as exc:  # noqa: BLE001 -- registra a falha sem abortar o manifesto
        return {"model_config_sha256": f"{PENDING} ({type(exc).__name__}: {exc})",
                "resolved_config_sha256": PENDING}


def clinvar_adapter_recipe(adapter_path: str) -> dict[str, Any]:
    """Extrai a receita do C do bundle (config), o que §4.2/§4.4 exigem identico em M1-M4."""
    try:
        import torch

        bundle = torch.load(adapter_path, map_location="cpu", weights_only=False)
        cfg = bundle.get("config", {})
        keys = ["model_family", "model_version", "regime", "loss_type", "lora_rank", "lora_alpha",
                "lora_dropout", "lr_backbone", "lr_head", "batch_size", "grad_accum_steps",
                "max_epochs", "focal_gamma", "freeze_backbone", "context_size", "head_type"]
        return {k: cfg.get(k) for k in keys if k in cfg}
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--splits-manifest", type=Path, required=True)
    p.add_argument("--clinvar-adapter", type=Path, required=True, help="best_model.pt do M0 (o C)")
    p.add_argument("--calibration", type=Path, required=True, help="calibration_m0.json")
    p.add_argument("--checkpoint", required=True, help="checkpoint R03 (local ou s3://)")
    p.add_argument("--fasta", type=Path, default=None)
    p.add_argument("--skip-fasta-hash", action="store_true", help="pula o sha256 do fasta (~3GB)")
    p.add_argument("--out", type=Path, required=True)
    # Metadados conhecidos (defaults da Fase 0; ajuste se preciso).
    p.add_argument("--checkpoint-name", default="LUM-20260719-001-R03")
    p.add_argument("--checkpoint-uri",
                   default="s3://croma-bioai-lumina-artifacts-us-east-2/experiments/LUM-20260719-001/runs/R03/checkpoints/final/best_checkpoint.pt")
    p.add_argument("--checkpoint-file-sha256", default="f2983560f8f965d234df4f4d15e81337b9547a578d4d6b397ab640207d6a2f44")
    p.add_argument("--clinvar-release", default="2026-06-06")
    p.add_argument("--clinvar-source-uri", default="s3://ai4bio-lumina/benchmarks/mosaic/data/raw/clinvar/clinvar_20260606.vcf.gz")
    p.add_argument("--master-uri", default="s3://ai4bio-lumina-experiments-v2/lumina-ssm/data/datasets/clinvar/regional_abraom/clinvar_regional_abraom_master.parquet")
    p.add_argument("--fasta-uri", default="~/hg38/hg38.fa")
    args = p.parse_args(argv)

    splits = json.loads(args.splits_manifest.read_text(encoding="utf-8"))
    s = splits.get("splits", {})
    t = splits.get("test_sets", {})

    fasta_sha = PENDING
    if args.fasta is not None and not args.skip_fasta_hash:
        print(f"hashing fasta {args.fasta} (~3GB, aguarde)...", flush=True)
        fasta_sha = sha256_file(args.fasta)

    contract: dict[str, Any] = {
        "campaign": "regionalizacao-R03 (M0-M4, Eduardo)",
        "phase_frozen": "Fase 0 (dados) + Fase 1 (M0/C). gnomAD/ABraOM-final e chr8 = Fase 2/5.",
        "foundation_checkpoint": {
            "name": args.checkpoint_name,
            "artifact_uri": args.checkpoint_uri,
            "file_sha256": args.checkpoint_file_sha256,
            **checkpoint_config_hashes(args.checkpoint),
        },
        "reference_genome": {
            "assembly": "GRCh38",
            "fasta_uri": args.fasta_uri,
            "fasta_sha256": fasta_sha,
        },
        "clinvar": {
            "release_date": args.clinvar_release,
            "source_uri": args.clinvar_source_uri,
            "preprocessing_version": f"master:{args.master_uri}",
        },
        "abraom": {
            "release_or_download_date": PENDING + " (Fase 2)",
            "source_uri": PENDING + " (Fase 2)",
            "preprocessing_version": PENDING + " (Fase 2)",
        },
        "gnomad": {
            "release": PENDING + " (Fase 2 - decisao Eduardo: estratificacao/versao/path)",
            "genome_or_exome": "genome+exome (respondido); path/estratificacao PENDING",
            "source_uri": PENDING + " (Fase 2)",
            "preprocessing_version": PENDING + " (Fase 2)",
        },
        "splits": {
            "salt": splits.get("salt"),
            "fracs": splits.get("fracs"),
            "train_manifest_sha256": s.get("train", {}).get("sha256", PENDING),
            "validation_manifest_sha256": s.get("validation", {}).get("sha256", PENDING),
            "calibration_manifest_sha256": s.get("calibration", {}).get("sha256", PENDING),
            "test_br_manifest_sha256": t.get("T_BR", {}).get("sha256", PENDING),
            "test_nonbr_manifest_sha256": t.get("T_nonBR_matched", {}).get("sha256", PENDING),
            "chr8_manifest_sha256": PENDING + " (Fase 5)",
            "counts": {
                "train": s.get("train", {}).get("n"), "validation": s.get("validation", {}).get("n"),
                "calibration": s.get("calibration", {}).get("n"),
                "T_BR": t.get("T_BR", {}).get("n"), "T_nonBR_matched": t.get("T_nonBR_matched", {}).get("n"),
            },
            "overlap_check": splits.get("overlap_check"),
        },
        "clinvar_adapter_C": {
            "note": "§4.2: o MESMO C (backbone R03 congelado + LoRA + head) e reusado em M0-M4.",
            "artifact": str(args.clinvar_adapter),
            "sha256": sha256_file(args.clinvar_adapter),
            "calibration_json": str(args.calibration),
            "calibration_sha256": sha256_file(args.calibration),
            "recipe": clinvar_adapter_recipe(str(args.clinvar_adapter)),
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    try:
        import yaml  # type: ignore

        text = yaml.safe_dump(contract, sort_keys=False, allow_unicode=True)
        out = args.out if args.out.suffix in (".yaml", ".yml") else args.out.with_suffix(".yaml")
    except Exception:  # noqa: BLE001 -- sem PyYAML -> JSON
        text = json.dumps(contract, indent=2, ensure_ascii=False)
        out = args.out.with_suffix(".json")
    out.write_text(text, encoding="utf-8")

    print(f"C sha256 = {contract['clinvar_adapter_C']['sha256']}")
    print(f"R03 model_config_sha256 = {contract['foundation_checkpoint']['model_config_sha256']}")
    print(f"R03 resolved_config_sha256 = {contract['foundation_checkpoint']['resolved_config_sha256']}")
    pend = [k for k in ("abraom", "gnomad") if PENDING in json.dumps(contract[k])]
    print(f"pendencias (Fase 2): {pend} + chr8 (Fase 5)")
    print(f"saved={out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
