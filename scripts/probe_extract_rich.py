#!/usr/bin/env python3
"""R1 -- extracao RICA: uma passada de GPU, tudo que as ablacoes do R2 precisam.

O forward e o caro; guardar mais coisa dele e quase gratis. Extraimos de uma vez as 27
profundidades do mid-stack, os registers, a representacao regional, o trunk pre e pos norma em tres
escalas, as cabecas supervisionadas e a media reverse-complement.

BATCH DE 4, COMO NA SONDA
    [ref, alt, rc(ref), rc(alt)] -- ref e alt no MESMO forward, que e a invariante medida na sonda
    (o vies de kernel por tamanho de batch cancela na diferenca; ver PROBE_BATCH_SIZE em extract.py).

QUATRO CHECAGENS EMBUTIDAS -- rodam nas primeiras variantes e param o run se falharem
    1. **Mapeamento full-res -> mid.** Assumimos que o focal ``f`` cai em ``f//4`` no mid, mas o
       downsample e conv k=4 s=2 p=1 duas vezes. Confirmamos pelo argmax de ||Delta|| ao longo do
       mid: se o pico nao estiver onde esperamos, o sweep de camada inteiro seria lixo.
    2. **Os registers respondem a variante?** Se ``Delta`` nos registers for ~0, eles nao carregam
       nada sobre o alelo (podem ainda servir de contexto) -- e nao vale guardar os 16.
    3. **Equivariancia RC.** Quanto o modelo alinha a sequencia e seu complemento reverso. Media RC
       so faz sentido se as duas representacoes vivem no mesmo referencial; abaixo de ~0.9 de
       cosseno, mediar seria somar vetores de bases diferentes.
    4. **fp16 basta?** Guardamos em fp16 (metade do disco). Reportamos o erro relativo de
       quantizacao sobre os valores reais -- se for grande perto do sinal, subimos para fp32.

USO
    export R03_CKPT="s3://croma-bioai-lumina-artifacts-us-east-2/experiments/LUM-20260719-001/runs/R03/checkpoints/final/best_checkpoint.pt"
    PYTHONPATH="$WORK" "$PY" scripts/probe_extract_rich.py \\
        --release-root ~/mosaic-v1 --fasta ~/hg38/hg38.fa --checkpoint "$R03_CKPT" \\
        --out-dir ~/probe/rich
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.embedding_probe.rich import (  # noqa: E402
    MidStackTaps,
    assert_r03_head_layout,
    head_readouts,
    mid_index,
    pooled,
    reverse_complement,
    substitution_onehot,
)
from eval.embedding_probe.windows import WindowError, build_window, focal_offset  # noqa: E402

DEFAULT_CKPT = (
    "s3://croma-bioai-lumina-artifacts-us-east-2/experiments/LUM-20260719-001/"
    "runs/R03/checkpoints/final/best_checkpoint.pt"
)
CONV_HORIZON, LOCAL_ATTENTION = 25, 512  # os horizontes medidos/derivados na sonda

# Um token do mid ve ~+-16 bp de entrada (stem +-7, depois 2x conv k=4 s=2), entao a base focal
# cai no campo receptivo de varios tokens -- nao de um so. Offsets de -4 a +4 sao geometricamente
# possiveis; o smoke mediu o pico de ||Delta|| em -1..+2 com MODA 0, ou seja o token mais central
# domina, como a formula f//4 preve. Agregamos essa vizinhanca em vez de pegar um token.
MID_SPAN = (-1, 2)          # tokens do mid agregados em torno de f//4
MID_OFFSET_LIMIT = 4        # geometricamente possivel; fora disso a indexacao estaria errada


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--release-root", type=Path, required=True)
    p.add_argument("--fasta", type=Path, required=True)
    p.add_argument("--checkpoint", default=DEFAULT_CKPT)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--window-bp", type=int, default=4096)
    p.add_argument("--tiers", default="gold", help="'gold' ou 'gold,consensus'")
    p.add_argument("--limit", type=int, default=0, help="0 = todas; >0 trunca (para testar)")
    p.add_argument("--registers-full", action="store_true",
                   help="guarda os 16 registers (6.144 dims) em vez de so media/maximo")
    p.add_argument("--check-n", type=int, default=32, help="variantes usadas nas checagens embutidas")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    import numpy as np
    import pandas as pd
    import torch
    from pyfaidx import Fasta

    from eval.embedding_probe.extract import ProbeModel

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tiers = tuple(t.strip() for t in args.tiers.split(","))
    ex = pd.read_parquet(args.release_root / "pb_examples.parquet")
    sel = ex[ex["label_tier"].isin(tiers) & ex["sequence_eligible"]].reset_index(drop=True)
    if args.limit:
        sel = sel.head(args.limit)
    print(f"[rich] {len(sel):,} variantes ({'+'.join(tiers)}, sequence_eligible) "
          f"· janela {args.window_bp} bp")

    fasta = Fasta(str(args.fasta), as_raw=True, sequence_always_upper=True)

    def fetch(chrom: str, start: int, end: int) -> str:
        if start < 0 or chrom not in fasta:
            return ""
        return str(fasta[chrom][start:end])

    probe = ProbeModel(args.checkpoint, args.device,
                       dtype=getattr(torch, args.dtype), allow_tf32=False)
    taps = MidStackTaps(probe.model)
    focal = focal_offset(args.window_bp)
    focal_mid = mid_index(focal)
    rc_focal = args.window_bp - 1 - focal
    d_model, n_reg, n_taps = probe.d_model, taps.n_registers, taps.n_taps
    print(f"[rich] {n_taps} tomadas do mid-stack · {n_reg} registers · d_model={d_model} "
          f"· focal={focal} (mid {focal_mid})")

    # As fatias dos configs enderecam colunas de heads_lin; um checkpoint com outro
    # num_counterfactual_effect_classes as desloca TODAS sem dar erro. Falhar aqui e barato.
    layout = assert_r03_head_layout(probe.model)
    print("[rich] layout de heads_lin conferido: "
          + " · ".join(f"{n.replace('_head', '')}[{a}:{b}]" for n, a, b in layout))

    # --------------------------------------------------------------------------------------
    blocks: dict[str, list] = {}
    ids, failures, checks = [], [], {"mid_argmax": [], "register_response": [], "rc_cosine": []}

    def add(name: str, vec) -> None:
        blocks.setdefault(name, []).append(vec)

    t0 = time.perf_counter()
    for i, row in enumerate(sel.itertuples()):
        try:
            w = build_window(fetch, chrom=row.chrom, pos_1based=int(row.pos_1based),
                             ref=row.ref, alt=row.alt, window_bp=args.window_bp)
        except WindowError as exc:
            failures.append({"variant_id": row.variant_id, "reason": str(exc)})
            continue

        taps.reset()
        seqs = [w.ref_seq, w.alt_seq, reverse_complement(w.ref_seq), reverse_complement(w.alt_seq)]
        # TODO o trabalho tensorial fica dentro do inference_mode: tensores criados nele carregam
        # restricoes (version counter, autograd) que so nao mordem enquanto o contexto esta aberto.
        with torch.inference_mode():
            pre, post = probe.encode(seqs)                       # [4, L, d_full]
            states = taps.states                                 # n_taps x [4, n_reg+L/4, d_model]
            h_up, h_pure = probe.split(pre)
            last = states[-1]
            # states[-1] E a saida do mid-stack (com os registers ainda prependados); o
            # mid_hidden_state do encode() e ele sem registers. Nao ha forward extra a fazer.

            # sweep de camada: empilha na GPU e transfere UMA vez. 27 transferencias separadas por
            # variante custariam ~500 mil sincronizacoes no run completo.
            lo, hi = n_reg + focal_mid + MID_SPAN[0], n_reg + focal_mid + MID_SPAN[1] + 1
            sweep = torch.stack([(st[1, lo:hi] - st[0, lo:hi]).mean(0)
                                 for st in states]).float().cpu().numpy()
            for d in range(n_taps):
                add(f"L{d:02d}", sweep[d])

            # --- trunk, nas tres escalas da arquitetura -----------------------------------
            dhu = (h_up[1] - h_up[0]).unsqueeze(0)               # [1, L, d_model]
            add("delta_focal", dhu[0, focal].float().cpu().numpy())
            add("delta_p25", pooled(dhu, focal, CONV_HORIZON)[0].float().cpu().numpy())
            add("delta_p512", pooled(dhu, focal, LOCAL_ATTENTION)[0].float().cpu().numpy())
            add("delta_p512_max",
                pooled(dhu, focal, LOCAL_ATTENTION, reduce="max")[0].float().cpu().numpy())
            add("ref_focal", h_up[0, focal].float().cpu().numpy())
            add("ref_p512", pooled(h_up[0:1], focal, LOCAL_ATTENTION)[0].float().cpu().numpy())
            add("post_delta", (post[1, focal] - post[0, focal]).float().cpu().numpy())
            add("post_ref", post[0, focal].float().cpu().numpy())
            add("hpure_delta", (h_pure[1, focal] - h_pure[0, focal]).float().cpu().numpy())

            # --- media RC: o focal vira L-1-focal no complemento reverso -------------------
            dhu_rc = h_up[3, rc_focal] - h_up[2, rc_focal]
            add("delta_focal_rc", dhu_rc.float().cpu().numpy())
            add("delta_focal_rcavg", ((dhu[0, focal] + dhu_rc) / 2).float().cpu().numpy())

            # --- mid e registers -----------------------------------------------------------
            add("mid_delta",
                (last[1, n_reg + focal_mid] - last[0, n_reg + focal_mid]).float().cpu().numpy())
            add("mid_ref", last[0, n_reg + focal_mid].float().cpu().numpy())
            reg_delta = last[1, :n_reg] - last[0, :n_reg]        # [n_reg, d_model]
            add("reg_delta_mean", reg_delta.mean(0).float().cpu().numpy())
            add("reg_delta_max", reg_delta.abs().amax(0).float().cpu().numpy())
            add("reg_ref_mean", last[0, :n_reg].mean(0).float().cpu().numpy())
            if args.registers_full:
                add("reg_all", reg_delta.reshape(-1).float().cpu().numpy())

            # --- cabecas supervisionadas ---------------------------------------------------
            hr = head_readouts(probe.model, post[0:1, focal], post[1:2, focal])
            for key, name in (("linear", "heads_lin"), ("mlp", "heads_mlp"), ("ref", "heads_ref")):
                add(name, hr[key][0].float().cpu().numpy())

            add("subst", np.array(substitution_onehot(row.ref, row.alt), dtype=np.float32))

            # --- checagens embutidas, nas primeiras variantes ------------------------------
            if len(checks["mid_argmax"]) < args.check_n:
                per_pos = (last[1] - last[0]).norm(dim=-1)       # [n_reg + L/4]
                checks["mid_argmax"].append(int(per_pos[n_reg:].argmax()) - focal_mid)
                focal_norm = float((h_up[1, focal] - h_up[0, focal]).norm())
                checks["register_response"].append(
                    float(reg_delta.norm(dim=-1).mean()) / max(focal_norm, 1e-9))
                a, b = h_up[0, focal].double(), h_up[2, rc_focal].double()
                checks["rc_cosine"].append(float(torch.dot(a, b) / (a.norm() * b.norm())))

        ids.append(row.variant_id)


        if (i + 1) % 500 == 0:
            rate = (time.perf_counter() - t0) / (i + 1)
            print(f"    {i + 1}/{len(sel)}  ({rate:.3f} s/variante, "
                  f"faltam ~{rate * (len(sel) - i - 1) / 60:.1f} min)", flush=True)

    tap_labels = taps.describe()
    taps.close()
    probe.close()
    elapsed = time.perf_counter() - t0
    print(f"[rich] {len(ids):,} variantes em {elapsed / 60:.1f} min ({len(failures)} falhas)")

    # --------------------------------------------------------------------------------------
    print("\n[rich] CHECAGENS")
    offs = checks["mid_argmax"]
    mode = max(set(offs), key=offs.count)
    # O criterio nao e "o pico esta sempre no mesmo lugar" (nao esta, e nao deveria: varios tokens
    # do mid veem a base focal). E "o token mais central domina, e nada cai fora do geometricamente
    # possivel" -- que e o que valida a formula f//4.
    mid_ok = abs(mode) <= 1 and all(abs(o) <= MID_OFFSET_LIMIT for o in offs)
    print(f"  1. mapeamento full-res -> mid: pico de ||Delta|| em {min(offs)}..{max(offs)}, "
          f"MODA {mode} (limite geometrico +-{MID_OFFSET_LIMIT}) -> "
          f"{'OK' if mid_ok else 'FORA DO ESPERADO'}")
    print(f"     sweep agrega os tokens {MID_SPAN[0]}..{MID_SPAN[1]} em torno de f//4")
    reg = checks["register_response"]
    reg_med = float(np.median(reg))
    print(f"  2. resposta dos registers: ||Delta_reg|| / ||Delta_focal|| mediana = {reg_med:.4f}"
          f" -> {'respondem' if reg_med > 0.01 else 'praticamente inertes'}")
    rc = checks["rc_cosine"]
    rc_med = float(np.median(rc))
    print(f"  3. equivariancia RC: cos(ref_fwd[f], ref_rc[L-1-f]) mediana = {rc_med:.4f}"
          f" -> {'mesmo referencial: media RC ok' if rc_med > 0.9 else 'referenciais DIFERENTES: preferir CONCATENAR (blocos delta_focal + delta_focal_rc) a mediar'}")

    stacked = {k: np.stack(v).astype(np.float32) for k, v in blocks.items()}
    ref_block = stacked["delta_focal"]
    q = ref_block.astype(np.float16).astype(np.float32)
    rel = float(np.abs(q - ref_block).max() / max(np.abs(ref_block).max(), 1e-9))
    print(f"  4. fp16: erro relativo maximo de quantizacao = {rel:.2e} -> "
          f"{'suficiente' if rel < 1e-2 else 'usar fp32'} (medido em delta_focal)")

    # fp16 satura em 65504. A checagem 4 mede so o delta_focal, cuja escala e ~1; ja heads_ref
    # guarda valores ABSOLUTOS de cabecas -- splice_distance_pred e uma distancia sem limite
    # superior conhecido. Um overflow viraria inf, o ridge propagaria NaN e o resultado sairia
    # como AUROC 0.5 sem nenhum aviso. Custa uma varredura conferir.
    estourou = []
    for name, arr in stacked.items():
        h = arr.astype(np.float16)
        if not np.isfinite(h).all():
            n = int((~np.isfinite(h)).sum())
            estourou.append(f"{name} ({n} valores, |max|={float(np.abs(arr).max()):.3e})")
    if estourou:
        raise SystemExit("  4b. fp16 SATUROU em: " + "; ".join(estourou)
                         + "\n      grave estes blocos em float32 antes de seguir")
    print(f"  4b. fp16: nenhum overflow em {len(stacked)} blocos "
          f"(|max| global = {max(float(np.abs(a).max()) for a in stacked.values()):.3e}, limite 6.55e+04)")

    out = {"variant_id": np.array(ids, dtype="U40")}
    total = 0
    for name, arr in stacked.items():
        out[f"blk_{name}"] = arr.astype(np.float16)
        total += arr.shape[1]
    npz = args.out_dir / "probe_features.npz"
    np.savez_compressed(npz, **out)

    manifest = {
        "n_variants": len(ids), "window_bp": args.window_bp, "focal_index": focal,
        "focal_mid": focal_mid, "tiers": list(tiers), "dtype_stored": "float16",
        "n_taps": n_taps, "tap_labels": tap_labels,
        "n_registers": n_reg, "total_dims": total, "minutes": elapsed / 60,
        "blocks": {name: int(a.shape[1]) for name, a in stacked.items()},
        "checks": {
            "mid_offset_range": [int(min(offs)), int(max(offs))], "mid_ok": bool(mid_ok),
            "register_response_median": reg_med,
            "rc_cosine_median": rc_med,
            "fp16_relative_error": rel,
        },
        "n_failures": len(failures), "failures": failures[:20],
    }
    (args.out_dir / "probe_features.manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    size_mb = npz.stat().st_size / 2**20
    print(f"\n[rich] {total:,} dims/variante · {size_mb:.0f} MB -> {npz}")
    if not mid_ok:
        print("[rich] PARE: o mapeamento para o mid nao bate; o sweep de camada seria invalido.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
