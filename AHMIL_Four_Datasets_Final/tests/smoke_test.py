"""Synthetic end-to-end smoke test; does not require public EEG data."""

from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import torch
from scipy.io import savemat

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ahmil.model import AHMIL, GaussianFuzzyLayer, ParameterMatchedMLP


def make_npz_dataset(folder: Path, num_subjects: int, num_classes: int,
                     instance_shape, multi_label: bool = False):
    rng = np.random.default_rng(1234)
    folder.mkdir(parents=True, exist_ok=True)
    for subject in range(num_subjects):
        data, labels, lengths = [], [], []
        for class_id in range(num_classes):
            for _ in range(5):
                count = 3
                data.append(rng.normal(size=(count, *instance_shape)).astype(np.float32))
                labels.extend([class_id] * count)
                lengths.append(count)
        x = np.concatenate(data, axis=0)
        y = np.asarray(labels, dtype=np.int64)
        if multi_label:
            y = np.stack((y, y, y), axis=1)
        np.savez(
            folder / f"{subject + 1:02d}.npz",
            data=x,
            labels=y,
            lengths=np.asarray(lengths, dtype=np.int64),
        )


def make_hbued_dataset(folder: Path, num_subjects: int = 3):
    rng = np.random.default_rng(5678)
    folder.mkdir(parents=True, exist_ok=True)
    for subject in range(1, num_subjects + 1):
        labels = []
        for class_id in [0, 1, 0, 1, 0, 1, 0, 1]:
            labels.extend([class_id] * 3)
        x = rng.normal(size=(24, 4, 32)).astype(np.float32)
        savemat(folder / f"DE_s{subject:02d}_stt.mat", {
            "rb_data": x,
            "data": x.copy(),
            "base_data": rng.normal(size=(8, 128)).astype(np.float32),
            "valence_labels": np.asarray(labels, dtype=np.int64).reshape(1, -1),
            "arousal_labels": np.asarray(labels, dtype=np.int64).reshape(1, -1),
        })


def run(command):
    import os
    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = "1"
    environment["MKL_NUM_THREADS"] = "1"
    result = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=120,
    )
    if result.returncode:
        raise RuntimeError(result.stdout)
    return result.stdout


def check_model_variants():
    for rules in [3, 5, 7, 9, 11]:
        fuzzy = GaussianFuzzyLayer(16, rules)
        mlp = ParameterMatchedMLP(16, rules)
        assert sum(p.numel() for p in fuzzy.parameters()) == sum(p.numel() for p in mlp.parameters())

    input_cases = [
        (62, "time_first", 3, (1, 2, 32, 62)),
        (62, "time_first", 4, (1, 2, 32, 62)),
        (14, "time_first", 2, (1, 2, 32, 14)),
        (4, "channels_first", 2, (1, 2, 4, 32)),
    ]
    for channels, layout, classes, shape in input_cases:
        for variant in ["baseline", "mlp", "fuzzy", "hyper", "fuzzy_epc", "full"]:
            model = AHMIL(
                input_channels=channels,
                num_classes=classes,
                input_layout=layout,
                embed_dim=16,
                variant=variant,
                curvature=1.5,
            )
            inputs = torch.randn(*shape)
            mask = torch.ones(shape[:2], dtype=torch.bool)
            logits, _ = model(inputs, mask)
            logits.mean().backward()
            assert logits.shape == (shape[0], classes)


def main():
    check_model_variants()
    with tempfile.TemporaryDirectory(prefix="ahmil_smoke_") as temporary:
        root = Path(temporary)
        datasets = {
            "seed": (3, (32, 62), False, "emotion"),
            "seed_iv": (4, (32, 62), False, "emotion"),
            "dreamer": (2, (32, 14), True, "arousal"),
        }
        for name, (classes, shape, multi_label, task) in datasets.items():
            folder = root / name
            make_npz_dataset(folder, 2, classes, shape, multi_label)
            output = root / f"out_{name}"
            command = [
                sys.executable, "-m", "scripts.train",
                "--dataset", name,
                "--data_dir", str(folder),
                "--task", task,
                "--output_dir", str(output),
                "--epochs", "1",
                "--patience", "1",
                "--batch_size", "8",
                "--num_workers", "0",
                "--embed_dim", "16",
                "--gpu", "-1",
            ]
            run(command)
            assert (output / "subject_results.csv").exists()
            assert (output / "confusion_matrix_raw.csv").exists()
            assert (output / "predictions.csv").exists()

        hbued = root / "hbued"
        make_hbued_dataset(hbued, num_subjects=2)
        output = root / "out_hbued"
        run([
            sys.executable, "-m", "scripts.train",
            "--dataset", "hbued",
            "--data_dir", str(hbued),
            "--task", "valence",
            "--output_dir", str(output),
            "--epochs", "1",
            "--patience", "1",
            "--batch_size", "8",
            "--num_workers", "0",
            "--embed_dim", "16",
            "--gpu", "-1",
        ])
        assert (output / "subject_results.csv").exists()
        assert (output / "confusion_matrix_raw.csv").exists()

        validation_output = root / "validation_only"
        run([
            sys.executable, "-m", "scripts.train",
            "--dataset", "seed",
            "--data_dir", str(root / "seed"),
            "--task", "emotion",
            "--output_dir", str(validation_output),
            "--epochs", "1",
            "--patience", "1",
            "--batch_size", "8",
            "--num_workers", "0",
            "--embed_dim", "16",
            "--gpu", "-1",
            "--validation_only",
        ])
        assert (validation_output / "subject_results.csv").exists()
        assert not (validation_output / "predictions.csv").exists()
        assert not (validation_output / "confusion_matrix_raw.csv").exists()

    print("AHMIL synthetic smoke tests passed for all four datasets.")


if __name__ == "__main__":
    main()
