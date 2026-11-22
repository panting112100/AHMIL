"""Dataset loading and trial-level bag construction for four EEG datasets."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Dataset

from data_readers import hbuedreader, dreamerreader, seedivreader, seedreader


@dataclass
class DatasetBundle:
    bags_by_subject: List[List[np.ndarray]]
    labels_by_subject: List[np.ndarray]
    subjects: List[str]
    input_channels: int
    input_layout: str
    num_classes: int
    class_names: List[str]
    task: str


class BagDataset(Dataset):
    def __init__(self, bags: Sequence[np.ndarray], labels: Sequence[int]):
        if len(bags) != len(labels):
            raise ValueError("The number of bags and labels must match.")
        self.bags = [torch.as_tensor(x.copy(), dtype=torch.float32) for x in bags]
        self.labels = torch.as_tensor(np.asarray(labels), dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.bags[index], self.labels[index]


class MILSequenceCollator:
    def __init__(self, pad_value: float = 0.0):
        self.pad_value = float(pad_value)

    def __call__(self, batch):
        bags, labels = zip(*batch)
        lengths = [int(bag.shape[0]) for bag in bags]
        if any(length <= 0 for length in lengths):
            raise ValueError("Every trial bag must contain at least one instance.")
        feature_shape = tuple(bags[0].shape[1:])
        if any(tuple(bag.shape[1:]) != feature_shape for bag in bags):
            raise ValueError("All instances in a batch must have the same feature shape.")

        max_length = max(lengths)
        padded = torch.full(
            (len(bags), max_length, *feature_shape),
            self.pad_value,
            dtype=torch.float32,
        )
        mask = torch.zeros((len(bags), max_length), dtype=torch.bool)
        for index, bag in enumerate(bags):
            padded[index, :lengths[index]] = bag
            mask[index, :lengths[index]] = True
        return padded, torch.stack(labels).long(), mask


def _segment_labels(labels: np.ndarray, task: str, dataset: str) -> np.ndarray:
    labels = np.asarray(labels)
    if dataset == "dreamer":
        task_index = dreamerreader.tasks[task]
        return labels[:, task_index].astype(np.int64)
    if dataset == "hbued":
        task_index = hbuedreader.tasks[task]
        return labels[:, task_index].astype(np.int64)
    return labels.astype(np.int64).reshape(-1)


def _to_trial_bags(data: np.ndarray, labels: np.ndarray,
                   lengths: np.ndarray, subject: str) -> Tuple[List[np.ndarray], np.ndarray]:
    data = np.asarray(data)
    labels = np.asarray(labels).reshape(-1)
    lengths = np.asarray(lengths, dtype=np.int64).reshape(-1)
    if int(lengths.sum()) != len(data) or len(labels) != len(data):
        raise ValueError(
            f"{subject}: data/label/length mismatch: "
            f"data={len(data)}, labels={len(labels)}, sum(lengths)={int(lengths.sum())}"
        )

    bags, bag_labels = [], []
    start = 0
    for trial_index, length in enumerate(lengths):
        end = start + int(length)
        bag_x = data[start:end]
        bag_y = labels[start:end]
        if len(bag_x) == 0:
            raise ValueError(f"{subject}: trial {trial_index} has no instances.")
        if not np.all(bag_y == bag_y[0]):
            raise ValueError(
                f"{subject}: trial {trial_index} contains inconsistent segment labels."
            )
        bags.append(bag_x.astype(np.float32, copy=False))
        bag_labels.append(int(bag_y[0]))
        start = end
    return bags, np.asarray(bag_labels, dtype=np.int64)


def load_dataset_bundle(dataset: str, data_dir: str, task: str, feature_key: str = "rb_data") -> DatasetBundle:
    dataset = dataset.lower().replace("-", "_")
    if dataset == "seed":
        data, labels, lengths, subjects = seedreader.load_dataset_preprocessed(data_dir)
        num_classes, input_channels, input_layout = 3, 62, "time_first"
        class_names = ["negative", "neutral", "positive"]
        if task not in {"emotion", "all"}:
            raise ValueError("SEED uses task='emotion'.")
        task = "emotion"
    elif dataset == "seed_iv":
        data, labels, lengths, subjects = seedivreader.load_dataset_preprocessed(data_dir)
        num_classes, input_channels, input_layout = 4, 62, "time_first"
        class_names = ["neutral", "sad", "fearful", "happy"]
        if task not in {"emotion", "all"}:
            raise ValueError("SEED-IV uses task='emotion'.")
        task = "emotion"
    elif dataset == "dreamer":
        if task not in {"valence", "arousal"}:
            raise ValueError("DREAMER task must be 'valence' or 'arousal'.")
        data, labels, lengths, subjects = dreamerreader.load_dataset_preprocessed(data_dir)
        num_classes, input_channels, input_layout = 2, 14, "time_first"
        class_names = ["low", "high"]
    elif dataset == "hbued":
        if task not in {"valence", "arousal"}:
            raise ValueError("HBUED task must be 'valence' or 'arousal'.")
        data, labels, lengths, subjects = hbuedreader.load_dataset(
            data_dir, feature_key=feature_key, expected_subjects=None, verbose=True
        )
        num_classes, input_channels, input_layout = 2, 4, "channels_first"
        class_names = ["low", "high"]
    else:
        raise ValueError("dataset must be one of: seed, seed_iv, dreamer, hbued")

    if not data:
        raise FileNotFoundError(f"No subjects were loaded from {data_dir}.")

    bags_by_subject, labels_by_subject = [], []
    for subject, x, y, trial_lengths in zip(subjects, data, labels, lengths):
        task_labels = _segment_labels(y, task, dataset)
        bags, bag_labels = _to_trial_bags(x, task_labels, trial_lengths, subject)
        bags_by_subject.append(bags)
        labels_by_subject.append(bag_labels)

    if dataset == "hbued":
        input_channels = int(data[0].shape[1])
    elif dataset == "dreamer":
        input_channels = int(data[0].shape[-1])
    else:
        input_channels = int(data[0].shape[-1])

    return DatasetBundle(
        bags_by_subject=bags_by_subject,
        labels_by_subject=labels_by_subject,
        subjects=list(subjects),
        input_channels=input_channels,
        input_layout=input_layout,
        num_classes=num_classes,
        class_names=class_names,
        task=task,
    )


def stratified_train_valid_split(bags: Sequence[np.ndarray], labels: np.ndarray,
                                 val_ratio: float, random_state: int):
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1.")
    labels = np.asarray(labels, dtype=np.int64)
    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=val_ratio, random_state=random_state
    )
    train_indices, valid_indices = next(
        splitter.split(np.zeros(len(labels), dtype=np.uint8), labels)
    )
    return (
        [bags[i] for i in train_indices], labels[train_indices],
        [bags[i] for i in valid_indices], labels[valid_indices],
    )


def inject_irrelevant_instances(bags: Sequence[np.ndarray], labels: Sequence[int],
                                ratio: float, rng: np.random.Generator) -> List[np.ndarray]:
    """Add donor instances so the irrelevant fraction of each resulting bag is ratio."""
    if not 0.0 <= ratio < 1.0:
        raise ValueError("interference_ratio must be in [0, 1).")
    copied = [np.asarray(bag, dtype=np.float32).copy() for bag in bags]
    labels = np.asarray(labels)
    if len(copied) != len(labels):
        raise ValueError("The number of bags and labels must match for instance injection.")
    if ratio == 0.0 or len(copied) < 2:
        return copied

    for index, bag in enumerate(copied):
        donor_indices = [j for j, other in enumerate(copied) if j != index and len(other) and labels[j] != labels[index]]
        if not donor_indices:
            donor_indices = [j for j, other in enumerate(copied) if j != index and len(other)]
        donor_bags = [copied[j] for j in donor_indices]
        if not donor_bags:
            continue
        count = int(np.ceil(len(bag) * ratio / (1.0 - ratio)))
        donors = []
        for _ in range(count):
            donor = donor_bags[int(rng.integers(0, len(donor_bags)))]
            donors.append(donor[int(rng.integers(0, len(donor)))])
        if donors:
            copied[index] = np.concatenate((bag, np.stack(donors).astype(np.float32)), axis=0)
    return copied
