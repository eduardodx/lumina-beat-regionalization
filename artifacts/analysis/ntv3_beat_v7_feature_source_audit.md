# NTv3 Beat-v7 Feature Source Audit

Date: 2026-05-21

## Conclusion

For the current `beat-v7-gated-hybrid-phylo-structure-seed0` run, the NTv3
functional head receives only the final per-token hidden state of the last
Beat-v7 layer, plus low-dimensional auxiliary predictions from internal Lumina
heads (`phylo100`, `phylo470`, `structure_logits`). It does not receive
intermediate layer states, U-Net skips, pooled features, or multi-resolution
features.

## Code Path

1. `FunctionalTracksModel.forward` calls `_functional_feature_bundle`.
2. With `feature_source="hidden"`, `_functional_feature_bundle` calls
   `_backbone_hidden_states`.
3. `_backbone_hidden_states` calls `backbone.encode(input_ids)` when available.
4. `DNAFoundationBeatV7.encode` returns the final `hidden` after all 8 BeatV7
   blocks, two local-attention residual insertions, final layer norm, and
   dropout.
5. The NTv3 head consumes that tensor.

Relevant files:

- `eval/ntv3/heads.py:32-47`
- `eval/ntv3/heads.py:119-164`
- `eval/ntv3/heads.py:459-468`
- `src/models/beat_v7/model.py:142-166`

## Available Feature Sources

The NTv3 config supports only:

- `hidden`
- `decoder`

For beat-v7, `hidden` is the valid option. `decoder` is a compatibility path for
architectures exposing `extract_sequence_features(...)["decoder_states"]` or
`outputs["decoder_states"]`. Beat-v7 does not expose decoder states.

Also, auxiliary features (`phylo`, `structure`, `phylo-structure`) explicitly
require `feature_source="hidden"`.

Relevant files:

- `eval/ntv3/config.py:219-220`
- `eval/ntv3/config.py:237-238`
- `eval/ntv3/run.py:292`

## Beat-v7 Architecture Shape

Beat-v7 in this repository is not a U-Net-style tower/transformer/upsample
architecture. It is a same-resolution sequence model:

- token embedding at 1bp/token
- 8 bidirectional Mamba3 blocks
- each block mixes forward and reverse-complement/reverse-direction streams
- local attention is inserted every 3 blocks, with window 256
- final layer norm + dropout
- no downsampling
- no upsampling
- no exposed skip tensors for NTv3

There are residual connections inside the blocks, and local-attention residual
adds inside `encode`, but those are internal to the backbone. The NTv3 head only
receives the final hidden state.

Relevant files:

- `src/models/beat_v7/model.py:22-43`
- `src/models/beat_v7/model.py:63-99`
- `src/models/beat_v7/model.py:151-166`
- `src/models/beat_shared.py:191-202`
- `src/models/beat_v7/local_attn.py:32-48`

## Forward Probe

Checkpoint audited:

`artifacts/analysis/ntv3_recent/beat-v7-gated-hybrid-phylo-structure-seed0/human/functional/best_model.pt`

Probe input:

- species: human
- split: validation
- batch size: 1
- sequence length: 32768
- target center fraction: 0.375
- device: CUDA

Observed tensors:

```text
input_ids                         [1, 32768]
token_embedding                   [1, 32768, 256]
block_0_output                    [1, 32768, 256]
block_1_output                    [1, 32768, 256]
block_2_output                    [1, 32768, 256]
attn_0_output_before_residual_add [1, 32768, 256]
block_3_output                    [1, 32768, 256]
block_4_output                    [1, 32768, 256]
block_5_output                    [1, 32768, 256]
attn_1_output_before_residual_add [1, 32768, 256]
block_6_output                    [1, 32768, 256]
block_7_output                    [1, 32768, 256]
feature_bundle_hidden             [1, 32768, 256]
feature_bundle_aux                [1, 32768, 5]
head_aux_projection               [1, 32768, 32]
head_input_after_aux_concat       [1, 32768, 288]
head_full_length_output_before_crop [1, 32768, 34]
head_cropped_output               [1, 12288, 34]
targets                           [1, 12288, 34]
```

The manual head output crop exactly matched `FunctionalTracksModel.forward`:

```text
forward_matches_manual_crop_max_abs_diff = 0.0
```

Detailed probe JSON:

`artifacts/analysis/ntv3_recent/beat-v7-gated-hybrid-phylo-structure-seed0/human/functional/feature_source_audit.json`

## Resolution

The representation delivered to the head is 1bp resolution:

- input length: 32768 tokens
- hidden length: 32768 positions
- full head output: 32768 positions
- benchmark target/output after center crop: 12288 positions

There is no pooling before the functional head.

## Receptive Field

Architecturally, each BeatBlock contains a forward Mamba3 mixer and a reverse
Mamba3 mixer, then fuses the two streams. This means the final hidden state can,
in principle, use information from both directions across the full 32kb sequence.
The local attention layers add explicit local windows of 256 bp, but they are not
the only context mechanism.

This audit does not empirically measure effective receptive field. It only
confirms that the architecture exposes same-resolution final hidden states, and
that those states are produced after full-sequence bidirectional Mamba scans.

## Interpretation

The current NTv3 adapter is not leaving obvious U-Net skip tensors or
intermediate depth tensors on the table, because beat-v7 does not expose them.
It also is not accidentally pooling or striding the representation. The head gets
a dense 1bp final hidden state.

Therefore, if the head saturates around the current performance band, the next
bottleneck is less likely to be a trivial adapter shape/resolution bug and more
likely to be one of:

- final-layer representation quality at this model scale
- insufficient specialization of the final hidden state for high dynamic-range
  functional tracks
- lack of explicit access to intermediate-depth features, which would require
  instrumenting/changing the backbone API rather than selecting an existing
  feature source
