#!/usr/bin/env python3
"""Diagnostico do portao `batch_independence` da Fase 2. Roda na GPU.

O smoke achou que a MESMA sequencia da resultados diferentes em batch de 1 e em batch de 4
(pre=2.6e-3, pos=1.7e-2), embora 4 copias identicas dentro de um batch deem linhas identicas e
duas chamadas iguais deem resultado identico. Duas hipoteses, com consequencias opostas:

  (A) NUMERICA -- cuDNN/cuBLAS/flash-attn/Mamba escolhem algoritmo (tiling, split-k, ordem de
      reducao) em funcao da dimensao de batch. Soma em float nao e associativa, entao a ordem
      muda o ultimo bit e o erro se amplifica ao longo de 20 blocos. Depende do TAMANHO do batch,
      nunca do conteudo das outras linhas.
  (B) ESTRUTURAL -- algum operador mistura linhas do batch. Isso invalidaria extrair as 4 bases
      juntas, e a sonda teria de rodar uma sequencia por forward (4x mais lento).

Cinco testes, nesta ordem:

  1. **Ambiente**: flash-attn esta instalado? o caminho com janela esta ativo? (principal suspeito
     de (A): kernels de flash attention variam a particao do trabalho com o batch.)
  2. **Invariancia de CONTEUDO** (decisivo A vs B): linha 0 em `[ref,A,C,G]` vs em `[ref,ref,ref,ref]`
     -- MESMO tamanho de batch, conteudo diferente. Identico => nao ha cross-talk => (A).
  3. **Varredura de TAMANHO**: a mesma sequencia em B=1,2,4,8 -- quantifica (A).
  4. **Estabilidade do DELTA** (o que decide o desenho): `Delta(ref->alt)` medido em B=2, B=4 e B=8.
     Como ref e alt vao no MESMO batch, um vies de kernel afeta os dois igualmente e cancela.
  5. **Escala**: o efeito de batch comparado ao proprio sinal da variante, no mesmo tensor.
     "1.7e-2" so significa alguma coisa contra o tamanho do ||Delta|| que queremos medir.

USO
---
    PYTHONPATH="$WORK" "$PY" scripts/probe_batch_diagnostic.py \
        --manifest ~/probe/probe_manifest.parquet --fasta ~/hg38/hg38.fa \
        --checkpoint "$R03_CKPT" --out ~/probe/probe_batch_diag.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.embedding_probe.windows import build_window  # noqa: E402

DEFAULT_CKPT = (
    "s3://croma-bioai-lumina-artifacts-us-east-2/experiments/LUM-20260719-001/"
    "runs/R03/checkpoints/final/best_checkpoint.pt"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--fasta", type=Path, required=True)
    p.add_argument("--checkpoint", default=DEFAULT_CKPT)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--window-bp", type=int, default=4096)
    p.add_argument("--device", default="cuda")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    import numpy as np
    import pandas as pd
    import torch

    from eval.embedding_probe.extract import BASES, ProbeModel
    from pyfaidx import Fasta

    fasta = Fasta(str(args.fasta), as_raw=True, sequence_always_upper=True)

    def fetch(chrom: str, start: int, end: int) -> str:
        if start < 0 or chrom not in fasta:
            return ""
        return str(fasta[chrom][start:end])

    row = pd.read_parquet(args.manifest).iloc[0]
    win = build_window(fetch, chrom=row["chrom"], pos_1based=int(row["pos_1based"]),
                       ref=row["ref"], alt=row["alt"], window_bp=args.window_bp)
    ref_seq, focal = win.ref_seq, win.focal_index
    ref_base = ref_seq[focal]
    alt_base = row["alt"]
    alt_seq = ref_seq[:focal] + alt_base + ref_seq[focal + 1:]
    four = [ref_seq[:focal] + b + ref_seq[focal + 1:] for b in BASES]
    print(f"[diag] {row['chrom']}:{row['pos_1based']} {ref_base}>{alt_base} | L={args.window_bp}")

    probe = ProbeModel(args.checkpoint, args.device, dtype=torch.float32, allow_tf32=False)
    report: dict = {"window_bp": args.window_bp, "variant": f"{row['chrom']}:{row['pos_1based']}"
                                                            f":{ref_base}>{alt_base}"}

    def enc(seqs: list[str]) -> tuple:
        pre, post = probe.encode(seqs)
        return (pre.detach().float().cpu().numpy().copy(),
                post.detach().float().cpu().numpy().copy())

    # --- 1. ambiente -----------------------------------------------------------------------
    print("\n[1] ambiente de kernels")
    import lumina.models.local_attn as la

    flash_mod = la._flash_attn_module is not None
    flash_fn = la._flash_attn_func is not None
    windowed = [
        bool(getattr(m, "_flash_attn_supports_window", False))
        for m in probe.model.modules() if isinstance(m, la.LocalWindowAttention)
    ]
    env = {
        "flash_attn_importable": flash_mod,
        "flash_attn_func_available": flash_fn,
        "local_attention_modules": len(windowed),
        "using_windowed_flash": sum(windowed),
        "torch": torch.__version__,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "device_name": torch.cuda.get_device_name(0) if args.device.startswith("cuda") else "cpu",
    }
    for k, v in env.items():
        print(f"    {k:<28} {v}")
    report["environment"] = env

    # --- 2. invariancia de conteudo (DECISIVO) ---------------------------------------------
    print("\n[2] invariancia de CONTEUDO (mesmo B=4, conteudo diferente nas outras linhas)")
    pre_mixed, post_mixed = enc([ref_seq] + four[1:])       # ref + 3 sequencias diferentes
    pre_same, post_same = enc([ref_seq] * 4)                # ref + 3 copias do proprio ref
    shuffled = [ref_seq, four[0], four[0], four[0]]         # ref + 3 copias de OUTRA sequencia
    pre_shuf, post_shuf = enc(shuffled)
    d_pre_a = float(np.abs(pre_mixed[0] - pre_same[0]).max())
    d_pre_b = float(np.abs(pre_mixed[0] - pre_shuf[0]).max())
    d_post_a = float(np.abs(post_mixed[0] - post_same[0]).max())
    content_ok = max(d_pre_a, d_pre_b, d_post_a) == 0.0
    print(f"    linha 0: mista vs 4x-ref   pre={d_pre_a:.3e} post={d_post_a:.3e}")
    print(f"    linha 0: mista vs outra    pre={d_pre_b:.3e}")
    print(f"    -> {'SEM cross-talk de conteudo: hipotese (A), numerica' if content_ok else 'HA cross-talk de CONTEUDO: hipotese (B), estrutural'}")
    report["content_invariance"] = {"mixed_vs_same_pre": d_pre_a, "mixed_vs_other_pre": d_pre_b,
                                    "mixed_vs_same_post": d_post_a, "ok": content_ok}

    # --- 3. varredura de tamanho ------------------------------------------------------------
    print("\n[3] varredura de TAMANHO de batch (mesma sequencia)")
    by_size = {}
    for size in (1, 2, 4, 8):
        pre, post = enc([ref_seq] * size)
        by_size[size] = (pre[0], post[0])
    base_pre, base_post = by_size[4]
    sizes = {}
    for size, (pre, post) in by_size.items():
        sizes[str(size)] = {"pre_vs_b4": float(np.abs(pre - base_pre).max()),
                            "post_vs_b4": float(np.abs(post - base_post).max())}
        print(f"    B={size}: max|diff| vs B=4  pre={sizes[str(size)]['pre_vs_b4']:.3e} "
              f"post={sizes[str(size)]['post_vs_b4']:.3e}")
    report["size_sweep"] = sizes

    # --- 4. estabilidade do DELTA (o que decide o desenho) ----------------------------------
    print("\n[4] estabilidade do DELTA ref->alt (ref e alt sempre no MESMO batch)")
    deltas = {}
    configs = {
        "B2": [ref_seq, alt_seq],
        "B4": four,
        "B8": four + four,
    }
    for name, seqs in configs.items():
        pre, _ = enc(seqs)
        ref_i = 0 if name == "B2" else BASES.index(ref_base)
        alt_i = 1 if name == "B2" else BASES.index(alt_base)
        deltas[name] = pre[alt_i, focal] - pre[ref_i, focal]
    ref_delta = deltas["B4"]
    ref_norm = float(np.linalg.norm(ref_delta))
    delta_report = {"delta_l2_at_B4": ref_norm}
    for name, delta in deltas.items():
        abs_diff = float(np.abs(delta - ref_delta).max())
        rel = float(np.linalg.norm(delta - ref_delta) / ref_norm) if ref_norm else float("nan")
        delta_report[name] = {"max_abs_vs_B4": abs_diff, "relative_l2_vs_B4": rel}
        print(f"    {name}: max|Delta - Delta_B4|={abs_diff:.3e}  ||diff||/||Delta||={rel:.3e}")
    print(f"    ||Delta(ref->alt)|| no focal (pre-norm, B=4) = {ref_norm:.4f}")
    report["delta_stability"] = delta_report

    # --- 5. escala: efeito de batch vs sinal da variante -------------------------------------
    print("\n[5] escala: efeito de batch vs sinal da variante (mesmo tensor pre-norm, no focal)")
    pre_b1, _ = enc([ref_seq])
    pre_b4, _ = enc(four)
    batch_effect = float(np.linalg.norm(pre_b1[0, focal] - pre_b4[BASES.index(ref_base), focal]))
    ratio = ref_norm / batch_effect if batch_effect else float("inf")
    print(f"    ||efeito de batch (B1 vs B4)|| = {batch_effect:.6f}")
    print(f"    ||sinal da variante||          = {ref_norm:.6f}")
    print(f"    razao sinal/efeito             = {ratio:.1f}x")
    report["scale"] = {"batch_effect_l2": batch_effect, "variant_signal_l2": ref_norm,
                       "signal_to_batch_ratio": ratio}

    probe.close()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(f"\n[diag] relatorio: {args.out}")

    print("\n" + "=" * 78)
    if not content_ok:
        print("VEREDITO: cross-talk ESTRUTURAL entre linhas do batch.")
        print("  A extracao tem que passar a UMA sequencia por forward (4x mais lento).")
        return 1
    worst = max(v["relative_l2_vs_B4"] for k, v in delta_report.items() if isinstance(v, dict))
    print("VEREDITO: sem cross-talk de conteudo -- o efeito depende so do TAMANHO do batch,")
    print("  ou seja, e escolha de algoritmo de kernel (numerica), nao mistura entre amostras.")
    print(f"  Erro relativo do Delta entre configuracoes de batch: {worst:.3e}")
    print(f"  Razao sinal/efeito-de-batch: {ratio:.1f}x")
    print("  => desenho seguro DESDE QUE o tamanho do batch seja fixo (a sonda usa 4 sempre)")
    print("     e que ref/alt sejam comparados sempre DENTRO do mesmo forward.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
