# NTv3 Target Transform Audit

Date: 2026-05-21

## Scope

Compare the local Lumina NTv3 functional target transform against the official
InstaDeep NTv3 fine-tuning notebook:

- Official notebook: `notebooks_tutorials/03_fine_tuning_posttrained_model_biwig.ipynb`
- Local implementation: `eval/ntv3/dataset.py`
- Task audited: human functional BigWig tracks

## Conclusion

The local target transform is conformant with the official notebook.

Both pipelines:

1. Read BigWig values into `(seq_len, num_tracks)`.
2. Replace NaNs with zero.
3. Crop targets to the center target fraction.
4. Divide each track by its metadata `mean`.
5. Apply soft clipping above scaled value `10.0`:

```python
torch.where(
    scaled > 10.0,
    2.0 * torch.sqrt(scaled * 10.0) - 10.0,
    scaled,
)
```

An exact numerical equivalence check on sample tensors produced:

```text
max_abs_diff 0.0
```

## Official Notebook Evidence

Relevant notebook lines:

- BigWig values are read, transposed, NaNs converted to zero, center-cropped,
  then transformed: lines 6420-6437.
- `create_targets_scaling_fn` uses `metadata_df["mean"]`, divides by means,
  and soft-clips above `10.0`: lines 6509-6536.
- Poisson-Multinomial loss is then applied to transformed targets: lines
  6865-6911.

## Local Pipeline Evidence

Relevant local lines:

- `FunctionalTargetsScaler.__call__`: `eval/ntv3/dataset.py:61-64`
- BigWig target read/crop/transform order: `eval/ntv3/dataset.py:478-493`
- Transform construction from track metadata means: `eval/ntv3/train.py:1010-1011`
- Poisson-Multinomial loss: `eval/ntv3/losses.py:15-34`

## Empirical Clip Audit

Computed on the first 128 human validation windows, after center crop, using
raw BigWig targets and official/local scaling.

```text
samples 128
positions_per_track 1572864
assay                 n_tracks  clip_frac   mean_scaled  mean_clipped  clipped/scaled  max_scaled_mean  max_scaled_max
ATAC-seq              5         0.00741920  1.107915     1.095217      0.988539        36.81            64.42
Histone ChIP-seq      4         0.00686216  1.192423     1.149456      0.963967        42.08            106.91
PRO-cap               10        0.00336024  2.481885     0.379181      0.152780        192378.40        367971.94
eCLIP                 10        0.04005171  2.241902     1.254112      0.559396        1860.03          4496.37
polyA plus RNA-seq    2         0.01268037  1.640291     0.754010      0.459681        1133.12          1622.23
total RNA-seq         3         0.02514627  1.551426     1.309425      0.844014        376.52           470.84
```

Top tracks by fraction of positions above the soft-clip threshold:

```text
ENCSR249ROI_M  eCLIP          clip_frac=0.08238157  max_scaled=1964.40
ENCSR154HRN_M  eCLIP          clip_frac=0.06341489  max_scaled=2709.43
ENCSR249ROI_P  eCLIP          clip_frac=0.06229591  max_scaled=325.27
ENCSR154HRN_P  eCLIP          clip_frac=0.05390167  max_scaled=767.98
ENCSR619DQO_M  total RNA-seq  clip_frac=0.03741709  max_scaled=470.84
ENCSR862QCH_M  eCLIP          clip_frac=0.03154627  max_scaled=3395.55
ENCSR484LTQ_M  eCLIP          clip_frac=0.02566910  max_scaled=4496.37
ENCSR321PWZ_M  eCLIP          clip_frac=0.02345403  max_scaled=1724.71
ENCSR321PWZ_P  eCLIP          clip_frac=0.02314568  max_scaled=1138.43
ENCSR410DWV    ATAC-seq       clip_frac=0.02018229  max_scaled=64.42
ENCSR619DQO_P  total RNA-seq  clip_frac=0.01996930  max_scaled=458.36
ENCSR862QCH_P  eCLIP          clip_frac=0.01825078  max_scaled=911.40
```

## Interpretation

The target transform is not an implementation divergence from the official
notebook. The threshold `10.0` is intentional in the reference notebook, not a
local Lumina deviation.

The empirical audit still shows that the transform materially compresses rare
high-intensity values, especially for eCLIP, polyA RNA-seq, total RNA-seq, and
some PRO-cap tracks. However, because the official notebook applies the same
transform, this compression is part of the benchmark fine-tuning recipe as
implemented by the public NTv3 tutorial.

Therefore, the current Lumina gap should not be attributed to an accidental
local target-transform mismatch.
