"""
HBUED MAT-file reader for cross-subject MIL experiments.

Expected MAT variables (confirmed from DE_s01_stt.mat):
    rb_data:        [N_segments, 4, 32]  baseline-corrected DE features
    data:            [N_segments, 4, 32]  original DE features
    base_data:       [N_trials, 128]
    valence_labels:  [1, N_segments] or [N_segments]
    arousal_labels:  [1, N_segments] or [N_segments]

The stored binary labels are used directly. No thresholding or relabeling is performed.
"""

import os
import re
import glob
import numpy as np
import scipy.io as sio

tasks = {"valence": 0, "arousal": 1}

def _subject_sort_key(path):
    name = os.path.basename(path)
    match = re.search(r"DE_s(\d+)", name, flags=re.IGNORECASE)
    return (0, int(match.group(1))) if match else (1, name.lower())

def _read_label(mat, key, mat_path):
    if key not in mat:
        raise KeyError(f"{os.path.basename(mat_path)} is missing variable {key!r}")
    label = np.asarray(mat[key]).reshape(-1)
    if label.size == 0:
        raise ValueError(f"{os.path.basename(mat_path)} {key} is empty")
    if not np.all(np.isfinite(label)):
        raise ValueError(f"{os.path.basename(mat_path)} {key} contains NaN/Inf")
    unique = np.unique(label)
    if not np.all(np.isin(unique, [0, 1])):
        raise ValueError(
            f"{os.path.basename(mat_path)} {key} must contain binary 0/1 labels; "
            f"observed values are {unique.tolist()}"
        )
    return label.astype(np.int64)

def load_subject_mat(mat_path, feature_key="rb_data", verbose=True):
    """Load one HBUED subject and return flattened segments, frame labels and trial lengths."""
    mat = sio.loadmat(mat_path)

    if feature_key not in mat:
        available = [k for k in mat.keys() if not k.startswith("__")]
        raise KeyError(
            f"{os.path.basename(mat_path)} is missing feature variable {feature_key!r}; "
            f"Available variables: {available}"
        )

    x = np.asarray(mat[feature_key], dtype=np.float32)
    if x.ndim != 3:
        raise ValueError(
            f"{os.path.basename(mat_path)} {feature_key} must be a 3-D array "
            f"[segments, 4, 32], actual shape is {x.shape}"
        )
    if not np.all(np.isfinite(x)):
        raise ValueError(f"{os.path.basename(mat_path)}  {feature_key}  contains NaN/Inf")

    valence = _read_label(mat, "valence_labels", mat_path)
    arousal = _read_label(mat, "arousal_labels", mat_path)

    n_segments = x.shape[0]
    if valence.size != n_segments or arousal.size != n_segments:
        raise ValueError(
            f"{os.path.basename(mat_path)} feature and label lengths differ: "
            f"features={n_segments}, valence={valence.size}, arousal={arousal.size}"
        )
    if "base_data" not in mat:
        raise KeyError(f"{os.path.basename(mat_path)} is missing base_data; trial count cannot be inferred")
    base_data = np.asarray(mat["base_data"])
    if base_data.ndim < 2:
        raise ValueError(
            f"{os.path.basename(mat_path)} has an invalid base_data shape: {base_data.shape}"
        )
    n_trials = int(base_data.shape[0])
    if n_trials <= 0 or n_segments % n_trials != 0:
        raise ValueError(
            f"{os.path.basename(mat_path)} cannot be partitioned into trials: "
            f"segments={n_segments}, trials={n_trials}"
        )

    segments_per_trial = n_segments // n_trials
    lengths = np.full(n_trials, segments_per_trial, dtype=np.int64)
    valence_by_trial = valence.reshape(n_trials, segments_per_trial)
    arousal_by_trial = arousal.reshape(n_trials, segments_per_trial)
    for task_name, task_labels in (
        ("valence", valence_by_trial),
        ("arousal", arousal_by_trial),
    ):
        inconsistent = np.where(np.any(task_labels != task_labels[:, :1], axis=1))[0]
        if inconsistent.size:
            raise ValueError(
                f"{os.path.basename(mat_path)}  {task_name} labels are inconsistent within trial; "
                f"invalid trial indices (zero-based): {inconsistent.tolist()}"
            )

    labels = np.stack([valence, arousal], axis=1).astype(np.int64)

    if verbose:
        trial_valence = valence_by_trial[:, 0]
        trial_arousal = arousal_by_trial[:, 0]
        v_counts = dict(zip(*np.unique(trial_valence, return_counts=True)))
        a_counts = dict(zip(*np.unique(trial_arousal, return_counts=True)))
        subject_match = re.search(r"(s\d+)", os.path.basename(mat_path), flags=re.IGNORECASE)
        subject_name = subject_match.group(1).lower() if subject_match else os.path.splitext(os.path.basename(mat_path))[0]
        print(
            f"[HBUED] {subject_name}: data={tuple(x.shape)}, "
            f"trials={n_trials}, segments/trial={segments_per_trial}, "
            f"valence={v_counts}, arousal={a_counts}"
        )

    return x, labels, lengths

def load_dataset(datapath, feature_key="rb_data", expected_subjects=None, verbose=True):
    """
    Load all HBUED subject MAT files.

    Returns:
        data:     list of arrays, each [total_segments, 4, 32]
        labels:   list of arrays, each [total_segments, 2] (valence, arousal)
        lengths:  list of arrays, each [n_trials], usually 24 entries of 60
        subjects: list of subject IDs, e.g. ['s01', 's02', ...]
    """
    mat_files = glob.glob(os.path.join(datapath, "DE_s*.mat"))
    mat_files = sorted(mat_files, key=_subject_sort_key)
    if not mat_files:
        raise FileNotFoundError(
            f"No DE_s*.mat files found in the HBUED directory: {datapath}"
        )

    if expected_subjects is not None and len(mat_files) != expected_subjects:
        raise ValueError(
            f"Expected {expected_subjects} subject files, found {len(mat_files)}."
            f"Directory: {datapath}"
        )

    data, labels, lengths, subjects = [], [], [], []
    reference_shape = None

    for mat_path in mat_files:
        x, y, trial_lengths = load_subject_mat(
            mat_path, feature_key=feature_key, verbose=verbose
        )
        if reference_shape is None:
            reference_shape = x.shape[1:]
        elif x.shape[1:] != reference_shape:
            raise ValueError(
                f"{os.path.basename(mat_path)} feature shape {x.shape[1:]} "
                f"differs from the first subject's shape {reference_shape}."
            )

        match = re.search(r"(s\d+)", os.path.basename(mat_path), flags=re.IGNORECASE)
        subject_name = match.group(1).lower() if match else os.path.splitext(os.path.basename(mat_path))[0]

        data.append(x)
        labels.append(y)
        lengths.append(trial_lengths)
        subjects.append(subject_name)

    if len(set(subjects)) != len(subjects):
        raise ValueError(f"Duplicate subject identifiers detected: {subjects}")

    if verbose:
        print(
            f"\n[HBUED] Loaded {len(subjects)} subjects; "
            f"feature_key={feature_key}; feature_shape={reference_shape}"
        )

    return data, labels, lengths, subjects

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Check and load HBUED MAT files")
    parser.add_argument("--data_dir", required=True, type=str)
    parser.add_argument("--feature_key", default="rb_data", choices=["rb_data", "data"])
    parser.add_argument("--expected_subjects", default=50, type=int)
    args = parser.parse_args()

    data, labels, lengths, subjects = load_dataset(
        args.data_dir,
        feature_key=args.feature_key,
        expected_subjects=args.expected_subjects,
        verbose=True,
    )
    print("\nData check passed.")
    print("subjects:", len(subjects))
    print("first subject:", subjects[0])
    print("first subject data:", data[0].shape)
    print("first subject labels:", labels[0].shape)
    print("first subject trial lengths:", lengths[0].shape, "unique=", np.unique(lengths[0]))
