"""SEED raw-data preprocessing and preprocessed subject-file loading."""

import argparse
import glob
import os
import re

import numpy as np
import scipy.io as sio
from scipy import interpolate
from scipy.signal import butter, sosfilt


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


def get_session_list(data_dir):
    sessions = []
    for path in glob.glob(os.path.join(data_dir, "*.mat")):
        if os.path.basename(path).lower() != "label.mat":
            sessions.append(os.path.splitext(os.path.basename(path))[0])
    return sorted(sessions)


def get_subject_list(data_dir):
    subjects = [session.split("_")[0] for session in get_session_list(data_dir)]
    return sorted(set(subjects), key=lambda value: (not value.isdigit(), int(value) if value.isdigit() else value))


def load_session_data(data_path, label_path, channels, target_classes=(0, 1, 2),
                      target_fs=128, filter_sos=None):
    data_dict = sio.loadmat(data_path)
    keys = [key for key in data_dict if not key.startswith("__")]
    keys.sort(key=lambda key: int(re.findall(r"\d+", key)[-1]) if re.findall(r"\d+", key) else 0)
    trials = [data_dict[key][channels, :].T for key in keys]

    original_labels = np.asarray(sio.loadmat(label_path)["label"]).reshape(-1).astype(np.int64)
    labels = original_labels + 1
    if len(trials) != len(labels):
        raise ValueError(
            f"Trial/label count mismatch in {os.path.basename(data_path)}: "
            f"{len(trials)} trials versus {len(labels)} labels."
        )

    processed = []
    for trial in trials:
        signal = trial
        if target_fs != 200:
            signal = resample(signal, 200, target_fs)
        if filter_sos is not None:
            signal = sosfilt(filter_sos, signal, axis=0)
        processed.append(signal.astype(np.float32, copy=False))

    keep = np.isin(labels, np.asarray(target_classes, dtype=np.int64))
    processed = [trial for trial, selected in zip(processed, keep) if selected]
    labels = labels[keep]
    if tuple(target_classes) == (0, 2):
        labels = labels // 2
    return processed, labels.tolist()


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
        raise ValueError("No EEG windows were generated; check the window and stride settings.")
    return (
        np.concatenate(segments, axis=0),
        np.asarray(segment_labels, dtype=np.int32),
        np.asarray(lengths, dtype=np.int64),
    )


def preprocess_subject(trials, labels, window, stride):
    return split_eegdata(trials, labels, window, stride)


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
    parser = argparse.ArgumentParser(description="Preprocess raw SEED sessions into trial bags.")
    parser.add_argument("--raw_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--window_sec", type=float, default=2.0)
    parser.add_argument("--stride_sec", type=float, default=2.0)
    parser.add_argument("--sampling_rate", type=int, default=128)
    parser.add_argument("--lowcut", type=float, default=0.3)
    parser.add_argument("--highcut", type=float, default=45.0)
    parser.add_argument("--classes", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()

    fs = args.sampling_rate
    window = int(args.window_sec * fs)
    stride = int(args.stride_sec * fs)
    filter_sos = butter(
        6, [args.lowcut, args.highcut], btype="bandpass", fs=fs, output="sos"
    )
    os.makedirs(args.output_dir, exist_ok=True)
    label_path = os.path.join(args.raw_dir, "label.mat")
    channels = np.arange(62)

    for subject in get_subject_list(args.raw_dir):
        all_data, all_labels, all_lengths = [], [], []
        for session in get_session_list(args.raw_dir):
            if not session.startswith(subject + "_"):
                continue
            trials, labels = load_session_data(
                os.path.join(args.raw_dir, session + ".mat"),
                label_path,
                channels,
                target_classes=args.classes,
                target_fs=fs,
                filter_sos=filter_sos,
            )
            data, trial_labels, lengths = preprocess_subject(trials, labels, window, stride)
            all_data.append(data)
            all_labels.append(trial_labels)
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
