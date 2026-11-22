"""SEED-IV raw-data preprocessing and preprocessed subject-file loading."""

import argparse
import glob
import os
import re

import numpy as np
import scipy.io as sio
from scipy import interpolate
from scipy.signal import butter, sosfilt


SEED_IV_LABELS = {
    "1": [1, 2, 3, 0, 2, 0, 0, 1, 0, 1, 2, 1, 1, 1, 2, 3, 2, 2, 3, 3, 0, 3, 0, 3],
    "2": [2, 1, 3, 0, 0, 2, 0, 2, 3, 3, 0, 1, 3, 1, 1, 2, 1, 0, 3, 0, 1, 3, 1, 2],
    "3": [1, 2, 2, 3, 0, 1, 2, 1, 2, 0, 1, 2, 3, 3, 0, 3, 0, 3, 1, 0, 1, 1, 3, 2],
}


def resample(signal, signal_frequency, target_frequency):
    ratio = float(signal_frequency) / float(target_frequency)
    source_positions = np.arange(len(signal))
    interpolator = interpolate.interp1d(
        source_positions, signal, axis=0, bounds_error=False, fill_value="extrapolate"
    )
    target_positions = np.arange(0, len(signal), ratio)
    expected_length = round(signal.shape[0] / signal_frequency * target_frequency)
    result = interpolator(target_positions)
    if len(result) < expected_length:
        padding = np.zeros((expected_length - len(result), signal.shape[-1]), dtype=result.dtype)
        result = np.concatenate((result, padding), axis=0)
    return result[:expected_length]


def get_session_list_seediv(data_dir):
    files = []
    for root, _, names in os.walk(data_dir):
        for name in names:
            if name.lower().endswith(".mat") and name.lower() != "label.mat":
                files.append(os.path.join(root, name))
    return sorted(files)


def get_subject_list(data_dir):
    subjects = set()
    for path in get_session_list_seediv(data_dir):
        name = os.path.basename(path)
        match = re.match(r"([^_]+)_", name)
        if match:
            subjects.add(match.group(1))
    return sorted(subjects, key=lambda value: (not value.isdigit(), int(value) if value.isdigit() else value))


def _session_id(path):
    parent = os.path.basename(os.path.dirname(path)).lower()
    if parent in {"1", "2", "3"}:
        return parent
    match = re.search(r"(?:session)?([123])", parent)
    if match:
        return match.group(1)
    raise ValueError(f"Cannot determine SEED-IV session number from directory: {parent}")


def load_session_data_seediv(data_path, target_fs=128, filter_sos=None):
    session_id = _session_id(data_path)
    trial_labels = SEED_IV_LABELS[session_id]
    data_dict = sio.loadmat(data_path)
    keys = [key for key in data_dict if not key.startswith("__")]
    keys.sort(key=lambda key: int(re.findall(r"\d+", key)[-1]) if re.findall(r"\d+", key) else 0)
    if len(keys) < 24:
        raise ValueError(
            f"Expected at least 24 trial variables in {os.path.basename(data_path)}, found {len(keys)}."
        )
    keys = keys[:24]

    trials = []
    for key in keys:
        signal = np.asarray(data_dict[key]).T.astype(np.float64, copy=False)
        if filter_sos is not None:
            signal = sosfilt(filter_sos, signal, axis=0)
        if target_fs != 200:
            signal = resample(signal, 200, target_fs)
        mean = np.mean(signal, axis=0, keepdims=True)
        std = np.std(signal, axis=0, keepdims=True)
        signal = (signal - mean) / (std + 1e-6)
        trials.append(signal.astype(np.float32, copy=False))
    return trials, trial_labels


def split_eegdata(trials, labels, window, stride):
    segments, segment_labels, lengths = [], [], []
    for trial, label in zip(trials, labels):
        trial_segments = []
        start = 0
        while start + window <= trial.shape[0]:
            trial_segments.append(trial[start:start + window])
            start += stride
        if not trial_segments:
            continue
        trial_segments = np.asarray(trial_segments, dtype=np.float32)
        segments.append(trial_segments)
        segment_labels.extend([int(label)] * len(trial_segments))
        lengths.append(len(trial_segments))
    if not segments:
        raise ValueError("No SEED-IV windows were generated; check the window and stride settings.")
    return (
        np.concatenate(segments, axis=0),
        np.asarray(segment_labels, dtype=np.int32),
        np.asarray(lengths, dtype=np.int64),
    )


def load_dataset_preprocessed(data_dir):
    files = glob.glob(os.path.join(data_dir, "*.npz"))
    def sort_key(path):
        name = os.path.basename(path)[:-4]
        return (0, int(name)) if name.isdigit() else (1, name.lower())
    files.sort(key=sort_key)
    if not files:
        raise FileNotFoundError(f"No subject NPZ files found in {data_dir}.")

    data, labels, lengths, subjects = [], [], [], []
    for path in files:
        with np.load(path, allow_pickle=False) as archive:
            data.append(archive["data"].astype(np.float32))
            labels.append(archive["labels"].astype(np.int64))
            lengths.append(archive["lengths"].astype(np.int64))
        subjects.append(os.path.basename(path)[:-4])
    return data, labels, lengths, subjects


def main():
    parser = argparse.ArgumentParser(description="Preprocess raw SEED-IV sessions into trial bags.")
    parser.add_argument("--raw_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--window_sec", type=float, default=2.0)
    parser.add_argument("--stride_sec", type=float, default=1.0)
    parser.add_argument("--sampling_rate", type=int, default=128)
    parser.add_argument("--lowcut", type=float, default=4.0)
    parser.add_argument("--highcut", type=float, default=45.0)
    args = parser.parse_args()

    fs = args.sampling_rate
    window = int(args.window_sec * fs)
    stride = int(args.stride_sec * fs)
    filter_sos = butter(
        6, [args.lowcut, args.highcut], btype="bandpass", fs=200, output="sos"
    )
    os.makedirs(args.output_dir, exist_ok=True)

    sessions = get_session_list_seediv(args.raw_dir)
    for subject in get_subject_list(args.raw_dir):
        all_data, all_labels, all_lengths = [], [], []
        for path in sessions:
            if not os.path.basename(path).startswith(subject + "_"):
                continue
            trials, labels = load_session_data_seediv(
                path, target_fs=fs, filter_sos=filter_sos
            )
            data, segment_labels, lengths = split_eegdata(trials, labels, window, stride)
            all_data.append(data)
            all_labels.append(segment_labels)
            all_lengths.append(lengths)

        if not all_data:
            continue
        output = os.path.join(args.output_dir, f"{subject}.npz")
        np.savez(
            output,
            data=np.concatenate(all_data).astype(np.float32),
            labels=np.concatenate(all_labels).astype(np.int32),
            lengths=np.concatenate(all_lengths).astype(np.int64),
        )
        print(f"Saved {output}")


if __name__ == "__main__":
    main()
