# AHMIL: Four-Dataset Cross-Subject EEG Emotion Recognition

This repository provides a paper-aligned implementation of AHMIL for SEED, SEED-IV, DREAMER, and HBUED.

## 1. Repository structure

```text
AHMIL_Four_Datasets_Final/
├── ahmil/
│   ├── model.py                 # MSIFE + FECA (FERA/PPEG) + HDM
│   ├── data.py                  # Dataset loading, trial bags, stratified split
│   └── engine.py                # LOSO training, validation selection, metrics
├── data_readers/
│   ├── seedreader.py
│   ├── seedivreader.py
│   ├── dreamerreader.py
│   └── hbuedreader.py
├── scripts/
│   ├── train.py                 # Main LOSO entry point
│   ├── run_multiseed.sh         # Seeds 42, 123, 2026
│   ├── run_ablation.sh          # Six model variants
│   ├── run_sensitivity.sh       # Fuzzy rules and curvature
│   ├── run_noise_robustness.sh  # Irrelevant-instance ratios
│   ├── run_electrode_occlusion.sh
│   └── paired_statistics.py
├── tests/
├── requirements.txt
├── PAPER_ALIGNMENT.md
└── .gitignore
```

## 2. Environment

Python 3.10 or newer is recommended. Install a PyTorch build matching the machine's CUDA version first, then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

The code does not include raw datasets, preprocessed arrays, checkpoints, or experiment outputs. PyTorch is intentionally omitted from `requirements.txt`; install a build compatible with your CUDA / CPU environment before installing the remaining dependencies.

## 3. Data formats

### SEED and SEED-IV

The preprocessed data directory contains one `.npz` file per subject. Each file must contain:

- `data`: concatenated EEG instances, shaped `[total_instances, time, 62]`.
- `labels`: one trial label repeated for each instance in that trial.
- `lengths`: number of instances in each trial.

Preprocess raw SEED data:

```bash
python data_readers/seedreader.py \
  --raw_dir /path/to/SEED/raw \
  --output_dir ./data/SEED/processed \
  --window_sec 2 --stride_sec 2
```

Preprocess raw SEED-IV data:

```bash
python data_readers/seedivreader.py \
  --raw_dir /path/to/SEED-IV/raw \
  --output_dir ./data/SEED-IV/processed \
  --window_sec 2 --stride_sec 1
```

The SEED-IV reader applies 4–45 Hz filtering at the original 200 Hz sampling rate, resamples to 128 Hz, and performs per-channel Z-score normalization within each trial before segmentation.

### DREAMER

Provide the original `DREAMER.mat` file. The preprocessor creates 23 subject-level NPZ files:

```bash
python data_readers/dreamerreader.py \
  --mat_path /path/to/DREAMER.mat \
  --output_dir ./data/DREAMER/processed \
  --window_sec 2 --stride_overlap 0
```

The reader stores valence, arousal, and dominance labels. Training selects valence or arousal using `--task`.

### HBUED

Place the 50 subject files (`DE_s01_stt.mat` through `DE_s50_stt.mat`) in one directory. By default, the reader uses `rb_data` shaped `[segments, 4, 32]`, `base_data` to infer the number of trials, and the stored binary `valence_labels` / `arousal_labels`.

Check the files before training:

```bash
python data_readers/hbuedreader.py --data_dir /path/to/HBUED --feature_key rb_data --expected_subjects 50
```

## 4. Main experiments

Run from the repository root. All four datasets use LOSO, trial-level stratified 80:20 training/validation splitting, validation-accuracy checkpoint selection, patience 50, 150 maximum epochs, and batch size 16.

```bash
# SEED: 15 folds, 3 classes
python -m scripts.train --dataset seed --data_dir ./data/SEED/processed \
  --task emotion --output_dir ./outputs/SEED_AHMIL

# SEED-IV: 15 folds, 4 classes
python -m scripts.train --dataset seed_iv --data_dir ./data/SEED-IV/processed \
  --task emotion --output_dir ./outputs/SEEDIV_AHMIL

# DREAMER: 23 folds, binary valence/arousal
python -m scripts.train --dataset dreamer --data_dir ./data/DREAMER/processed \
  --task valence --output_dir ./outputs/DREAMER_Valence
python -m scripts.train --dataset dreamer --data_dir ./data/DREAMER/processed \
  --task arousal --output_dir ./outputs/DREAMER_Arousal

# HBUED: 50 folds, binary valence/arousal
python -m scripts.train --dataset hbued --data_dir ./data/HBUED \
  --task valence --output_dir ./outputs/HBUED_Valence
python -m scripts.train --dataset hbued --data_dir ./data/HBUED \
  --task arousal --output_dir ./outputs/HBUED_Arousal
```

Dataset-specific optimizer defaults follow the manuscript table:

| Dataset | Optimizer | Learning rate | Weight decay |
|---|---|---:|---:|
| SEED | AdamW | 0.001 | 0.0001 |
| SEED-IV | AdamW | 0.001 | 0.0001 |
| DREAMER | Adam | 0.001 | 0.001 |
| HBUED | Adam | 0.0001 | 0.0001 |

The default learning-rate schedule is cosine annealing with 10 warm-up epochs and a minimum learning rate of `1e-6`.

## 5. Ablations and additional experiments

### Model variants

The `--variant` argument supports:

- `baseline`: RetMIL baseline, without fuzzy enhancement or HDM.
- `mlp`: parameter-matched MLP control replacing the Gaussian Fuzzy Layer.
- `fuzzy`: Baseline + Gaussian Fuzzy Layer.
- `hyper`: Baseline + Lorentz HDM.
- `fuzzy_epc`: Gaussian Fuzzy Layer + Euclidean prototype classifier.
- `full`: Gaussian Fuzzy Layer + Lorentz HDM (AHMIL).

Run all six variants:

```bash
bash scripts/run_ablation.sh seed ./data/SEED/processed emotion ./outputs/SEED_ablation
```

### Multi-seed experiments

```bash
bash scripts/run_multiseed.sh seed ./data/SEED/processed emotion ./outputs/SEED_multiseed
```

The script runs seeds 42, 123, and 2026. The held-out subject order and fold-wise train/validation partitions are fixed across these seeds (`--split_seed 42` by default); only model initialization and training randomness change.

### Fuzzy-rule and curvature sensitivity

```bash
bash scripts/run_sensitivity.sh seed ./data/SEED/processed emotion ./outputs/SEED_sensitivity
```

The fuzzy-rule values are `{3, 5, 7, 9, 11}` and curvature values are `{0.5, 1.0, 1.5, 2.0, 2.5}`. Sensitivity runs use `--validation_only`: they record the best validation accuracy per fold and do not evaluate held-out test subjects during hyperparameter selection.

### Irrelevant-instance robustness

```bash
bash scripts/run_noise_robustness.sh seed ./data/SEED/processed emotion ./outputs/SEED_noise
```

The irrelevant-instance fraction is measured relative to the resulting bag size and is set to 0%, 25%, 50%, and 75%. The same random seed gives deterministic donor selection for paired Baseline / Baseline + Fuzzy runs.

### Electrode occlusion

```bash
bash scripts/run_electrode_occlusion.sh seed ./data/SEED/processed emotion ./outputs/SEED_occlusion
```

The best validation-selected checkpoint is evaluated on the held-out subject with the original input and with one electrode zeroed at a time. The contribution score is `full_test_accuracy - occluded_test_accuracy`. The script saves subject-level scores, electrode-wise mean and standard deviation, and pairwise subject Spearman correlations. For HBUED, the electrode axis is the final dimension of the DE feature tensor.

## 6. Output files

Each run writes:

- `subject_results.csv`: per-held-out-subject best validation epoch and test metrics.
- `predictions.csv`: per-trial true label, predicted label, and class probabilities.
- `confusion_matrix_raw.csv`: raw counts summed across LOSO folds.
- `confusion_matrix_row_normalized.csv`: row-normalized matrix (rows=true classes, columns=predicted classes).
- `summary.json`: mean and standard deviation across held-out subjects, plus aggregated confusion matrices.
- `checkpoints/`: best validation-selected checkpoint for each fold.

## 7. Statistical analysis across seeds

After running multiple seeds for two methods, compare subject-level results after averaging each subject's scores across seeds:

```bash
python -m scripts.paired_statistics \
  --method_a ./outputs/SEED_AHMIL_multiseed \
  --method_b ./outputs/SEED_baseline_multiseed \
  --metrics test_accuracy balanced_accuracy macro_f1 \
  --output ./outputs/paired_accuracy.csv
```

The script reports subject-level paired mean differences, 95% confidence intervals, two-sided paired t-tests, and Holm-adjusted p-values across the requested metrics.

## 8. Code smoke test

Run the synthetic test suite without downloading the EEG datasets:

```bash
python tests/smoke_test.py
```

It checks all six model variants, verifies the parameter count of the MLP control against the Gaussian Fuzzy Layer, and executes short synthetic LOSO runs for all four dataset input formats.

## 9. Important reproducibility note

The package has been organized to follow the supplied manuscript and removes the extra ChannelAttention / instance-normalization wrappers, dataset-specific fuzzy warm-up, manual DREAMER fuzzy initialization, class weighting, label smoothing, and default PSEMix/CutMix augmentation that were not specified in the manuscript. It also standardizes the four dataset runners around the same LOSO and validation-only checkpoint-selection procedure.

The manuscript does not specify every low-level implementation constant. See `PAPER_ALIGNMENT.md` for the items that remain implementation defaults and should be checked against the exact historical experiment configuration before claiming numerical reproduction. The full 15/15/23/50-fold experiments have not been run in this packaging step, so the code is syntax-checked and smoke-tested rather than verified against the published accuracy values.
