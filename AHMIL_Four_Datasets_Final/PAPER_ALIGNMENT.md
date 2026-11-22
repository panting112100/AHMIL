# Paper-to-code alignment notes

This file records which implementation choices are explicitly supported by the supplied manuscript and which remain defaults because the manuscript does not state their exact values.

## Directly specified by the manuscript

| Manuscript item | Implementation |
|---|---|
| MSIFE has three parallel branches | `MultiScaleInstanceEncoder` has three branches |
| MSIFE kernel sizes are 3, 11, 25 | `branch_3`, `branch_11`, `branch_25` |
| Convolution channels are 16 / 32 | Each branch uses 16 then 32 output channels |
| FECA contains two FERA layers with PPEG between them | `FERA -> PPEG -> FERA` |
| FERA uses Gaussian membership, rule recombination, and residual enhancement | `GaussianFuzzyLayer` implements the membership and residual equations without an extra BatchNorm layer |
| Fuzzy rule count is 3 by default | `--fuzzynum 3` |
| Retention uses 8 heads | `--num_heads 8` |
| PPEG depthwise kernels are 7, 5, 3 | `PPEG` uses depthwise 1-D convolutions with those kernels |
| HDM maps to Lorentz space and uses one prototype per class | `LorentzPrototypeClassifier` stores one spatial prototype per class and derives its time coordinate |
| Default curvature is 1 | `--curvature 1.0` |
| SEED / SEED-IV / DREAMER / HBUED LOSO folds are 15 / 15 / 23 / 50 | Subject-wise LOSO over all loaded subjects |
| Training / validation split is 80:20 at trial level | `StratifiedShuffleSplit` is applied after excluding the held-out subject |
| Early stopping and checkpoint selection use validation accuracy; patience 50 | Implemented in `ahmil.engine.run_loso` |
| Maximum epochs 150; batch size 16 | Runner defaults |
| Optimizers and learning rates / weight decay | Dataset-specific defaults in `ahmil.engine.DEFAULTS` |
| Cosine schedule, 10 warm-up epochs, minimum LR `1e-6` | Implemented in `adjust_learning_rate` |
| Confusion matrices aggregate raw fold counts before row normalization | Raw counts are summed from all held-out predictions, then row-normalized |
| Multi-seed values are 42, 123, 2026 | `scripts/run_multiseed.sh`; fold-wise train/validation splits use a fixed `split_seed` independent of the model seed, so all three runs use identical partitions |
| Fuzzy-rule sensitivity values are 3, 5, 7, 9, 11 | `scripts/run_sensitivity.sh` uses validation-only model selection and does not evaluate held-out test subjects |
| Curvature sensitivity values are 0.5, 1.0, 1.5, 2.0, 2.5 | `scripts/run_sensitivity.sh` uses validation-only model selection and does not evaluate held-out test subjects |
| Irrelevant-instance ratios are 0%, 25%, 50%, 75% | `scripts/run_noise_robustness.sh` |
| Electrode attribution is based on single-electrode occlusion on held-out subjects | `--electrode_occlusion` |

## Dataset preprocessing details

- **SEED:** 2-second windows with a 2-second stride; labels are mapped from `{-1, 0, 1}` to `{0, 1, 2}`. The supplied reader defaults to 128 Hz and 0.3–45 Hz filtering. The manuscript text does not state the exact SEED filter band or target sampling rate; confirm these defaults against the preprocessing used for the reported results.
- **SEED-IV:** 4–45 Hz filtering at 200 Hz, resampling to 128 Hz, per-channel Z-score normalization within each trial, 2-second windows, and 1-second stride. Labels are neutral, sad, fearful, and happy.
- **DREAMER:** 2-second non-overlapping windows, 256 samples per instance, and separate binary valence / arousal tasks. The supplied reader also applies 4–30 Hz filtering, baseline correction, and per-trial channel standardization; the manuscript excerpt specifies the window and segment length but does not detail these three preprocessing operations.
- **HBUED:** each trial contains 60 DE-feature segments; each subject has 24 trials; the reader uses the stored binary valence / arousal labels and defaults to the baseline-corrected `rb_data` feature variable.

## Implementation defaults not numerically specified in the manuscript

The following values are explicit in the code but are not fully specified in the supplied manuscript text. They should be checked against the exact experiment configuration used to produce the paper's tables and figures:

1. Gradient clipping max norm is set to 1.0. The manuscript states that gradient clipping is used but does not give the threshold.
2. The Gaussian fuzzy residual gain `alpha` is initialized to zero. The manuscript describes it as learnable but does not state its initial value.
3. The learnable Lorentz-logit scale is initialized to 2.5. The manuscript defines the scale factor but does not state its initialization.
4. The retention-head decay coefficients follow the geometric schedule used in the supplied RetMIL implementation. The manuscript gives the decay form but not the complete head-wise initialization schedule.
5. MSIFE dropout rates follow the supplied implementation (0.25 in temporal blocks and 0.5 after branch fusion); these rates are not listed in the manuscript's parameter table.
6. FERA applies pre-LayerNorm before fuzzy-enhanced retention, following the RetMIL block structure; the manuscript equations do not separately state this normalization placement.
7. The tangent projection in HDM is initialized to the identity matrix with zero bias; the manuscript does not specify this initialization.

## Deliberately excluded from the paper-aligned release

The previous uploaded code bundle contained dataset-specific additions not described in the supplied manuscript. They are not included in this release's main model path:

- SEED-IV `ChannelAttention` and `NormalizedBackbone` wrapper.
- HBUED `ChannelAttention` and extra instance-normalization wrapper.
- DREAMER manual fuzzy-center / fuzzy-scale initialization.
- HBUED fuzzy-alpha warm-up and classifier learning-rate multiplier.
- Dataset-specific class weighting, label smoothing, and default PSEMix / CutMix augmentation.
- XPOS and extra GroupNorm / fuzzy BatchNorm operations not present in the manuscript equations.

The cleaned release therefore follows the method and evaluation protocol described in the supplied manuscript. Because these changes alter the implementation relative to the legacy bundle, the reported paper metrics must be rechecked by running the complete experiments before the repository is presented as an exact numerical reproduction.
