# Lumina: Practical Research Roadmap

## Framing

Lumina should not try to be, in one paper, a new genomic foundation model, a ClinVar paper, a Brazilian population study, and a MinION clinical pipeline.

The program is stronger if it is sequenced deliberately:

- first establish a reproducible compact biological pretraining core
- then make the model genuinely sensitive to the causal ref->alt change
- then benchmark clinical utility with explicit counterfactual design
- then evaluate calibration and failure modes on Brazilian or Latin American cohorts
- only after that, build translational laboratory workflow claims

This roadmap is written around that order.

## Research Objective

**To develop a compact, biologically supervised, allele-sensitive genomic foundation model for clinically relevant noncoding and splice-associated variant interpretation, and to validate it with rigorous counterfactual, temporal, and regional evaluation.**

## Publication Program

### Paper 1: Methods

**A compact, allele-sensitive genomic foundation model for noncoding and splice-relevant variant interpretation.**

Primary contribution:

- dense biological supervision
- compact reproducible baseline
- counterfactual ref/alt modeling
- strong held-out and counterfactual evaluation

### Paper 2: Regional Validation

**Brazilian-aware and admixed-population validation, calibration, and failure analysis.**

Primary contribution:

- temporal evaluation
- external calibration and subgroup analysis
- regional failure modes and VUS triage

### Paper 3: Translational Pipeline

**An end-to-end laboratory interpretation workflow that connects sequencing, calling, and model-guided analysis.**

Primary contribution:

- operational workflow integration
- turnaround-time and cost analysis
- deployment tradeoffs in the UFG context

## Guiding Principles

1. **Reproducibility before novelty.** A clean environment must run the code and produce the same baseline behavior before new research ideas are layered on top.
2. **Dense supervision before harder objectives.** Use all valid sequence positions for PhyloP and splice learning, while keeping MLM masked-only.
3. **Allele sensitivity before stronger downstream claims.** A model that performs well on clinical benchmarks without responding to the actual ref->alt change is not solving the right problem.
4. **Evaluation before optimization.** A held-out validation protocol is required before any claim about better sampling, better losses, or better long-context behavior.
5. **One major variable at a time.** New training objectives and architectural changes should be added as controlled ablations, not stacked all at once.
6. **Clinical and regional claims only after benchmark evidence.** Latin American relevance is a strong thesis, but it must be demonstrated with explicit benchmark design, calibration, and comparison.

---

## Phase 0: Reproducible Research Core [CRITICAL]

**Status:** COMPLETE

**Goal:** Make the repository reliably runnable and measurable before changing the science.

**Why this is needed:** If installation and execution are fragile, every later experiment becomes difficult to trust.

**Required work:**
- Align `pyproject.toml` and `requirements.txt` so the declared environment matches the code that is actually imported.
- Provide one supported install path and one supported smoke-test path.
- Remove module-level dataset and dataloader instantiation from `src/train.py` and `src/sanity.py` so imports do not trigger heavy data loading.
- Extract shared constants such as vocabulary IDs and structure labels into a single shared module.
- Ensure output directories, config serialization, and checkpointing behave consistently across runs.
- Add minimal run documentation so a collaborator can reproduce a sanity check and a short debug training run without guessing.

**Verification gate:** A clean environment can run the sanity check and a short training job end-to-end without manual fixes, missing packages, or import-time data side effects.

---

## Phase 1: Correct Objective Semantics [CRITICAL]

**Status:** COMPLETE

**Goal:** Make the training objective scientifically sound before making it more sophisticated.

**Why this is needed:** MLM is naturally sparse and corruption-based; PhyloP and splice labels are dense and should supervise the model everywhere they are valid.

**Required work:**
- Keep **MLM** as a masked-token objective only.
- Change **PhyloP100**, **PhyloP470**, and **splice structure** losses so they are computed on **all valid genomic positions**, not just on `mask_positions`.
- Define a clear validity mask for auxiliary tasks, excluding padding and ambiguous `N` positions.
- Normalize each auxiliary loss by the number of supervised positions so scale does not drift with sequence composition.
- Add imbalance handling for splice prediction, since dense all-position supervision will contain far more background than splice-positive bases.
- Report per-task loss magnitudes separately so the team can understand whether the dense auxiliary tasks dominate or vanish.
- Defer learned uncertainty-based loss weighting until the dense-objective baseline is stable.

**Verification gate:** Over 500 to 1,000 steps, all losses remain finite and comparable, and the dense auxiliary setup improves held-out PhyloP correlation and splice metrics relative to the masked-only auxiliary baseline.

---

## Phase 2: Data Correctness and Sampling Hygiene [CRITICAL]

**Status:** COMPLETE

**Goal:** Ensure the model is trained and evaluated on a trustworthy data stream.

**Why this is needed:** Biased or noisy sampling can make training appear better than it is, while invalid evaluation splits can make the model look stronger than it really is.

**Required work:**
- Replace uniform chromosome sampling with chromosome-length-weighted sampling.
- Introduce an explicit train/validation chromosome split so evaluation is not performed on the same sampling pool used for training.
- Replace the current splice interval lookup with a faster indexed structure so large-scale training remains feasible.
- Filter or downweight windows with excessive `N` content so centromeric and low-information regions do not dominate sampling.
- Log basic batch composition statistics such as chromosome frequency, `N` fraction, splice-positive density, and exon/regulatory overlap.
- Verify coordinate integrity across FASTA, BigWig, and GTF inputs before starting long training runs.

**Verification gate:** The loader produces stable, correctly distributed samples; validation is truly held out; and splice-label generation is fast enough not to become a bottleneck.

---

## Phase 3: Strong Compact Baseline at Fixed Length [HIGH]

**Status:** COMPLETE

**Goal:** Establish one trustworthy reference model before any downstream interface or architectural claims.

**Why this is needed:** The project needs a baseline that is small, repeatable, and cheap enough to rerun frequently. This is the anchor for every later comparison.

**Required work:**
- Standardize the first real baseline around the **Lumina-8M** tier.
- Train first at a fixed context length, preferably 4k once the data pipeline is stable.
- Add core training observability: MLM accuracy, PhyloP correlation, splice precision/recall/F1, gradient norm, tokens/sec, and validation curves.
- Integrate experiment tracking so runs are comparable across changes.
- Save enough metadata with checkpoints to make restarts and comparisons reproducible.
- Repeat baseline runs more than once to estimate variance.

**Verification gate:** The compact baseline trains reproducibly across multiple runs, improves on held-out metrics, and does not show obvious instability or metric drift.

---

## Phase 4: Counterfactual Variant Modeling and Allele Sensitivity [CRITICAL NEXT]

**Goal:** Make the model respond to the causal allelic change rather than relying mainly on genomic context priors.

**Why this is needed:** A model can score well on pathogenicity-style benchmarks while remaining weakly sensitive to the actual ref->alt edit. If that happens, the benchmark story can be stronger than the biology.

**Required work:**
- Introduce paired **reference and alternative** windows for downstream variant modeling.
- Build a downstream interface that uses **local token-level features** around the edited position, such as `h_ref`, `h_alt`, `delta`, and `abs(delta)`, plus wider context when justified.
- Add explicit **ref<->alt swap tests** to check whether the scorer tracks the direction of the mutation rather than just locus identity.
- Add **same-locus benign/pathogenic** comparisons so the model cannot rely on coarse context shortcuts.
- Compare local token-level clinical interfaces against frozen global pooled embeddings, and treat pooled embeddings as a side evaluation rather than the default clinical interface.
- Define allele-sensitivity diagnostics that can be run before large benchmark campaigns.

**Verification gate:** The model and downstream scorer show measurable sensitivity to the actual allelic edit under controlled same-locus and swap-based tests.

---

## Phase 5: Controlled Objective and Interface Ablations [HIGH]

**Goal:** Test important ideas one at a time and keep only what is justified.

**Why this is needed:** The project contains several plausible hypotheses, but they should not all enter the system at once. The model needs controlled comparisons so the team can identify what is truly helping.

**Required work:**
- Compare **dense auxiliary loss on all valid positions** versus auxiliary loss only on masked positions.
- Compare fixed task weights versus learned uncertainty-based task weighting.
- Compare the current cosine reverse-complement loss versus a contrastive RC objective.
- Compare 100% `MASK` replacement versus a BERT-style corruption mix.
- Run a major ablation on **discriminative all-position objectives** such as RTD/ELECTRA-style replaced-base detection versus pure MLM.
- Compare global pooled variant interfaces against local token-level delta interfaces under the same benchmark protocol.
- Keep each experiment isolated so conclusions remain interpretable.

**Verification gate:** Only changes that improve held-out biological metrics, allele sensitivity, stability, or downstream utility are promoted into the default training recipe.

---

## Phase 6: Clinical Benchmarking and Mechanism-Aware Evaluation [CRITICAL]

**Goal:** Translate the foundation model into a credible variant interpretation benchmark story.

**Why this is needed:** "Clinical significance" is too heterogeneous to support one vague evaluation claim. A serious methods paper needs explicit benchmark design, fair baselines, and consequence-aware reporting.

**Required work:**
- Build a reproducible **ClinVar + ABraOM-compatible** benchmark preparation pipeline.
- Define transparent variant inclusion rules, label policy, reference normalization, and allele-frequency criteria.
- Create benchmark splits that support zero-shot scoring, adapter-based adaptation, and full fine-tuning where appropriate.
- Include **temporal evaluation** so older variants are hidden during development and newer variants are reserved for testing.
- Report metrics separately for at least:
  - splice-proximal variants
  - noncoding regulatory or UTR or deep intronic variants
  - coding variants when included
- Use frozen probe, adapter or LoRA, full fine-tune, and zero-shot scoring as distinct evaluation regimes.
- Compare against explicit baselines and distinguish carefully between `NTv3_pre`, `NTv3_post`, and other models with additional post-training supervision.
- Prefer one benchmark family per regime rather than chasing every available benchmark.

**Verification gate:** The benchmark is reproducible, consequence-aware, and strong enough to support a methods-paper claim without leaning on vague pooled-embedding results.

---

## Phase 7: Efficient Sampling and Length Scaling [IMPORTANT]

**Goal:** Improve training efficiency and extend context only after the baseline and variant interface are trustworthy.

**Why this is needed:** Smart sampling and curriculum learning are promising, but they should optimize an already valid system, not rescue an unvalidated one.

**Required work:**
- Introduce the **Smart Sampler** as an explicit second-stage optimization, not the default from day one.
- Prioritize exons, promoters, and other high-information regions only after the unbiased baseline has been measured.
- Implement a length curriculum from **4k -> 16k -> 64k**, with the constant-token rule to keep memory and optimization stable.
- Add gradient accumulation if needed so the effective batch size remains meaningful as context grows.
- Monitor VRAM, throughput, and validation quality across curriculum transitions.
- Verify that the Mamba implementation follows the intended efficient path at longer lengths.

**Verification gate:** Context scaling improves efficiency and/or validation quality without collapsing the short-context baseline or introducing major instability.

---

## Phase 8: Advanced Architecture Only with Evidence [IMPORTANT]

**Goal:** Introduce architectural complexity only when simpler baselines are saturated.

**Why this is needed:** RoPE, fine-to-coarse hierarchy, and auxiliary-feature reinjection are all plausible, but they substantially increase implementation complexity and debugging burden. They should be treated as evidence-driven upgrades, not default next steps.

**Required work:**
- Add **RoPE** only if ablations show a real deficiency in positional anchoring that the baseline cannot solve.
- Prototype **fine-to-coarse pooling** only after 64k training is stable.
- Evaluate whether higher-level pooled representations actually help long-range genomic tasks without harming local motif and splice learning.
- Delay more speculative ideas such as auxiliary-feature reinjection into upper layers until a strong plain baseline exists.
- Use synthetic and biologically motivated long-range tests to justify each addition.

**Verification gate:** Advanced architecture changes improve long-range behavior while preserving or improving local biological metrics and allele-sensitive behavior.

---

## Phase 9: Regional Validation and Calibration [CRITICAL, BUT DOWNSTREAM]

**Goal:** Test whether the model remains well calibrated and clinically useful for Brazilian or admixed Latin American settings.

**Why this is needed:** The strongest defensible regional claim is not that current models are universally unusable, but that Brazilian and admixed Latin American populations remain under-evaluated, under-calibrated, and underrepresented in genomic foundation-model validation.

**Required work:**
- Assemble external holdout resources for Brazilian or Latin American validation where licensing and governance permit.
- Measure calibration, subgroup performance, and failure modes rather than reporting only aggregate AUROC or AUPRC.
- Evaluate whether regional adaptation helps on genuinely regional validation sets rather than anecdotal examples.
- Analyze VUS prioritization or triage behavior separately from binary pathogenicity classification.
- Document which gains come from better allele modeling, which come from calibration, and which come from adaptation.

**Verification gate:** Regional claims are backed by explicit external validation and calibration analysis rather than narrative alone.

---

## Phase 10: Translational Pipeline and Clinical Usability [FINAL]

**Goal:** Make the model explainable and deployable once it has earned that effort.

**Why this is needed:** MinION integration, interpretability, and laboratory deployment matter, but they should be built on top of a model whose predictive behavior is already credible.

**Required work:**
- Implement attribution or explanation methods for validated variant-scoring outputs.
- Check whether saliency maps highlight biologically plausible sequence features rather than artifacts.
- Evaluate lightweight deployment strategies only after core benchmark quality is acceptable.
- Build laboratory-facing workflow prototypes only after the model’s outputs and limitations are well understood.
- Separate model-quality claims from operational workflow claims in publications and reporting.

**Verification gate:** Interpretability outputs are biologically defensible, deployment tradeoffs do not materially damage benchmark performance, and translational claims are supported by actual workflow evaluation.

---

## Summary of Priority Order

1. Reproducibility and environment stability.
2. Correct loss semantics, especially dense auxiliary supervision at every valid position.
3. Data correctness, sampling hygiene, and held-out evaluation.
4. A repeatable compact baseline with proper metrics.
5. Counterfactual variant modeling and allele-sensitivity validation.
6. Controlled ablations for objectives and downstream interfaces.
7. Clinical benchmarking with consequence-aware and temporal evaluation.
8. Smart sampling and long-context curriculum.
9. Advanced architecture only if evidence demands it.
10. Regional validation and calibration.
11. Translational deployment and laboratory workflow integration.

---

## Non-Negotiable Success Criteria

- The code runs reproducibly in a clean environment.
- Validation is held out and not recycled from training.
- MLM is masked-only, but biological auxiliary losses supervise all valid positions.
- `N` bases are excluded from meaningful supervision.
- Metrics go beyond raw loss and reflect biological usefulness.
- Allele sensitivity is measured explicitly rather than assumed from benchmark scores.
- New ideas are adopted only after ablations show real value.
- Clinical claims are backed by explicit benchmarks, not by architecture alone.
- Regional claims are backed by external validation and calibration, not by narrative alone.
