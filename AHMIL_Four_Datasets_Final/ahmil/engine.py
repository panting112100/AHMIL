"""LOSO training, validation-based checkpoint selection, and test evaluation."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from ahmil.data import (
    BagDataset,
    DatasetBundle,
    MILSequenceCollator,
    inject_irrelevant_instances,
    stratified_train_valid_split,
)
from ahmil.model import AHMIL


DEFAULTS = {
    "seed": {"lr": 1e-3, "weight_decay": 1e-4, "optimizer": "adamw"},
    "seed_iv": {"lr": 1e-3, "weight_decay": 1e-4, "optimizer": "adamw"},
    "dreamer": {"lr": 1e-3, "weight_decay": 1e-3, "optimizer": "adam"},
    "hbued": {"lr": 1e-4, "weight_decay": 1e-4, "optimizer": "adam"},
}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def adjust_learning_rate(optimizer, epoch: int, args) -> float:
    warmup = max(int(args.warmup_epochs), 0)
    base_lr = float(args.lr)
    min_lr = float(args.min_lr)
    if warmup > 0 and epoch < warmup:
        lr = max(min_lr, base_lr * (epoch + 1) / warmup)
    else:
        decay_epochs = max(int(args.epochs) - warmup, 1)
        progress = min(max((epoch - warmup) / decay_epochs, 0.0), 1.0)
        lr = min_lr + (base_lr - min_lr) * 0.5 * (1.0 + np.cos(np.pi * progress))
    for group in optimizer.param_groups:
        group["lr"] = lr
    return float(lr)


def _make_loader(bags, labels, batch_size, shuffle, num_workers, device):
    dataset = BagDataset(bags, labels)
    generator = torch.Generator()
    generator.manual_seed(torch.initial_seed() % (2 ** 32))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=MILSequenceCollator(),
        generator=generator if shuffle else None,
    )


def _build_model(bundle: DatasetBundle, args) -> AHMIL:
    return AHMIL(
        input_channels=bundle.input_channels,
        num_classes=bundle.num_classes,
        input_layout=bundle.input_layout,
        embed_dim=args.embed_dim,
        num_rules=args.fuzzynum,
        curvature=args.curvature,
        variant=args.variant,
        num_heads=args.num_heads,
    )


def _forward_batch(batch, model, device):
    x, y, mask = batch
    x = x.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)
    mask = mask.to(device, non_blocking=True)
    logits, _ = model(x, mask)
    return logits, y


def train_one_epoch(loader, model, criterion, optimizer, device, grad_clip: float):
    model.train()
    losses, targets, predictions = [], [], []
    for batch in loader:
        logits, target = _forward_batch(batch, model, device)
        loss = criterion(logits, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        targets.append(target.detach().cpu().numpy())
        predictions.append(logits.argmax(dim=1).detach().cpu().numpy())
    y_true = np.concatenate(targets)
    y_pred = np.concatenate(predictions)
    return float(np.mean(losses)), 100.0 * accuracy_score(y_true, y_pred)


@torch.no_grad()
def predict(loader, model, device):
    model.eval()
    targets, predictions, probabilities = [], [], []
    for batch in loader:
        logits, target = _forward_batch(batch, model, device)
        targets.append(target.cpu().numpy())
        predictions.append(logits.argmax(dim=1).cpu().numpy())
        probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(targets), np.concatenate(predictions), np.concatenate(probabilities)


def _metrics(y_true, y_pred, y_prob):
    result = {
        "accuracy": 100.0 * accuracy_score(y_true, y_pred),
        "balanced_accuracy": 100.0 * balanced_accuracy_score(y_true, y_pred),
        "macro_f1": 100.0 * f1_score(y_true, y_pred, average="macro", zero_division=0),
    }
    try:
        if y_prob.shape[1] == 2:
            result["macro_auc"] = 100.0 * roc_auc_score(y_true, y_prob[:, 1])
        else:
            result["macro_auc"] = 100.0 * roc_auc_score(
                y_true, y_prob, multi_class="ovr", average="macro"
            )
    except ValueError:
        result["macro_auc"] = float("nan")
    return result


def _evaluate_bags(bags, labels, model, args, device):
    loader = _make_loader(
        bags, labels, args.batch_size, False, args.num_workers, device
    )
    y_true, y_pred, y_prob = predict(loader, model, device)
    return _metrics(y_true, y_pred, y_prob), y_true, y_pred, y_prob


def _run_electrode_occlusion(subject: str, bags: Sequence[np.ndarray], labels: np.ndarray,
                             model, args, device, full_accuracy: float):
    n_electrodes = int(bags[0].shape[-1])
    rows = []
    for electrode in range(n_electrodes):
        occluded = [bag.copy() for bag in bags]
        for bag in occluded:
            bag[..., electrode] = 0.0
        metrics, _, _, _ = _evaluate_bags(occluded, labels, model, args, device)
        rows.append({
            "subject": subject,
            "electrode": electrode,
            "full_accuracy": full_accuracy,
            "occluded_accuracy": metrics["accuracy"],
            "contribution": full_accuracy - metrics["accuracy"],
        })
    return rows


def _save_electrode_outputs(output_dir: Path, rows: List[dict]):
    if not rows:
        return
    subject_df = pd.DataFrame(rows)
    subject_df.to_csv(output_dir / "electrode_subject_contributions.csv", index=False)
    summary = subject_df.groupby("electrode")["contribution"].agg(["mean", "std"]).reset_index()
    summary["std"] = summary["std"].fillna(0.0)
    summary.to_csv(output_dir / "electrode_contribution_mean_std.csv", index=False)

    subject_vectors = {
        subject: group.sort_values("electrode")["contribution"].to_numpy()
        for subject, group in subject_df.groupby("subject")
    }
    subjects = list(subject_vectors)
    pair_rows = []
    for i, subject_a in enumerate(subjects):
        for subject_b in subjects[i + 1:]:
            rho = spearmanr(subject_vectors[subject_a], subject_vectors[subject_b]).statistic
            pair_rows.append({"subject_a": subject_a, "subject_b": subject_b, "spearman_rho": float(rho)})
    pair_df = pd.DataFrame(pair_rows)
    pair_df.to_csv(output_dir / "electrode_pairwise_spearman.csv", index=False)
    if len(pair_df):
        pd.DataFrame([{
            "mean_spearman_rho": pair_df["spearman_rho"].mean(),
            "std_spearman_rho": pair_df["spearman_rho"].std(ddof=0),
            "num_subject_pairs": len(pair_df),
        }]).to_csv(output_dir / "electrode_stability_summary.csv", index=False)


def run_loso(bundle: DatasetBundle, args) -> Dict[str, object]:
    dataset_key = args.dataset.lower().replace("-", "_")
    defaults = DEFAULTS[dataset_key]
    if args.lr is None:
        args.lr = defaults["lr"]
    if args.weight_decay is None:
        args.weight_decay = defaults["weight_decay"]
    if args.optimizer is None:
        args.optimizer = defaults["optimizer"]

    if args.interference_ratio < 0 or args.interference_ratio >= 1:
        raise ValueError("interference_ratio must be in [0, 1).")
    if args.validation_only and args.electrode_occlusion:
        raise ValueError("Electrode occlusion requires test evaluation and cannot be combined with --validation_only.")
    if args.epochs < 1 or args.patience < 1 or args.batch_size < 1:
        raise ValueError("epochs, patience, and batch_size must be positive.")

    if torch.cuda.is_available() and args.gpu >= 0:
        if args.gpu >= torch.cuda.device_count():
            raise ValueError(f"GPU {args.gpu} is unavailable.")
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")

    output_dir = Path(args.output_dir).resolve()
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    fold_rows, prediction_rows, electrode_rows = [], [], []
    all_true, all_pred = [], []

    for fold, test_subject in enumerate(bundle.subjects):
        fold_seed = int(args.seed) + fold
        set_seed(fold_seed)
        print(f"\n[{fold + 1}/{len(bundle.subjects)}] Test subject: {test_subject}")

        test_bags = bundle.bags_by_subject[fold]
        test_labels = bundle.labels_by_subject[fold]
        trainval_bags, trainval_labels = [], []
        for subject_index, (subject_bags, subject_labels) in enumerate(
            zip(bundle.bags_by_subject, bundle.labels_by_subject)
        ):
            if subject_index == fold:
                continue
            trainval_bags.extend(subject_bags)
            trainval_labels.extend(subject_labels.tolist())
        trainval_labels = np.asarray(trainval_labels, dtype=np.int64)

        train_bags, train_labels, valid_bags, valid_labels = stratified_train_valid_split(
            trainval_bags,
            trainval_labels,
            val_ratio=args.val_ratio,
            random_state=int(args.split_seed) + fold,
        )

        if args.interference_ratio > 0:
            train_bags = inject_irrelevant_instances(
                train_bags, train_labels, args.interference_ratio,
                np.random.default_rng(fold_seed + 11)
            )
            valid_bags = inject_irrelevant_instances(
                valid_bags, valid_labels, args.interference_ratio,
                np.random.default_rng(fold_seed + 23)
            )
            test_bags = inject_irrelevant_instances(
                test_bags, test_labels, args.interference_ratio,
                np.random.default_rng(fold_seed + 37)
            )

        train_loader = _make_loader(
            train_bags, train_labels, args.batch_size, True, args.num_workers, device
        )
        valid_loader = _make_loader(
            valid_bags, valid_labels, args.batch_size, False, args.num_workers, device
        )

        model = _build_model(bundle, args).to(device)
        if args.optimizer == "adamw":
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=args.lr, weight_decay=args.weight_decay
            )
        elif args.optimizer == "adam":
            optimizer = torch.optim.Adam(
                model.parameters(), lr=args.lr, weight_decay=args.weight_decay
            )
        else:
            raise ValueError("optimizer must be 'adamw' or 'adam'.")

        criterion = nn.CrossEntropyLoss()
        best_val_accuracy = -float("inf")
        best_epoch = 0
        patience_count = 0
        best_checkpoint = checkpoint_dir / f"fold_{fold + 1:02d}_best.pth"

        for epoch in range(args.epochs):
            adjust_learning_rate(optimizer, epoch, args)
            train_loss, train_accuracy = train_one_epoch(
                train_loader, model, criterion, optimizer, device, args.grad_clip
            )
            valid_true, valid_pred, _ = predict(valid_loader, model, device)
            valid_accuracy = 100.0 * accuracy_score(valid_true, valid_pred)

            if valid_accuracy > best_val_accuracy:
                best_val_accuracy = valid_accuracy
                best_epoch = epoch + 1
                patience_count = 0
                torch.save(model.state_dict(), best_checkpoint)
            else:
                patience_count += 1

            if (epoch + 1) % args.log_interval == 0 or epoch == 0:
                print(
                    f"Epoch {epoch + 1:03d} | train loss={train_loss:.4f} "
                    f"| train acc={train_accuracy:.2f}% | val acc={valid_accuracy:.2f}% "
                    f"| best val={best_val_accuracy:.2f}%"
                )
            if patience_count >= args.patience:
                break

        if not best_checkpoint.exists():
            raise RuntimeError(f"No validation checkpoint was saved for {test_subject}.")
        model.load_state_dict(torch.load(best_checkpoint, map_location=device, weights_only=True))

        if args.validation_only:
            fold_rows.append({
                "subject": test_subject,
                "best_epoch": best_epoch,
                "best_val_accuracy": best_val_accuracy,
            })
            print(f"Validation-only {test_subject}: best val acc={best_val_accuracy:.2f}% (epoch {best_epoch})")
            del model, optimizer, train_loader, valid_loader
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

        test_metrics, y_true, y_pred, y_prob = _evaluate_bags(
            test_bags, test_labels, model, args, device
        )
        row = {
            "subject": test_subject,
            "best_epoch": best_epoch,
            "best_val_accuracy": best_val_accuracy,
            "test_accuracy": test_metrics["accuracy"],
            "balanced_accuracy": test_metrics["balanced_accuracy"],
            "macro_f1": test_metrics["macro_f1"],
            "macro_auc": test_metrics["macro_auc"],
        }
        fold_rows.append(row)
        all_true.extend(y_true.tolist())
        all_pred.extend(y_pred.tolist())
        for sample_index, (true_label, pred_label, probability) in enumerate(
            zip(y_true, y_pred, y_prob)
        ):
            prediction_rows.append({
                "subject": test_subject,
                "trial_index": sample_index,
                "y_true": int(true_label),
                "y_pred": int(pred_label),
                **{f"prob_class_{k}": float(value) for k, value in enumerate(probability)},
            })

        print(
            f"Test {test_subject}: acc={test_metrics['accuracy']:.2f}% "
            f"BA={test_metrics['balanced_accuracy']:.2f}% "
            f"Macro-F1={test_metrics['macro_f1']:.2f}% "
            f"(best epoch {best_epoch})"
        )

        if args.electrode_occlusion:
            electrode_rows.extend(_run_electrode_occlusion(
                test_subject, test_bags, test_labels, model, args, device,
                test_metrics["accuracy"]
            ))

        del model, optimizer, train_loader, valid_loader
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    results = pd.DataFrame(fold_rows)
    results.to_csv(output_dir / "subject_results.csv", index=False)
    if args.validation_only:
        summary = {
            "dataset": dataset_key,
            "task": bundle.task,
            "variant": args.variant,
            "seed": int(args.seed),
            "validation_only": True,
            "subjects": len(bundle.subjects),
            "mean_best_validation_accuracy": float(results["best_val_accuracy"].mean()),
            "std_best_validation_accuracy": float(results["best_val_accuracy"].std(ddof=0)),
            "fuzzynum": int(args.fuzzynum),
            "curvature": float(args.curvature),
        }
        with open(output_dir / "summary.json", "w", encoding="utf-8") as file:
            json.dump(summary, file, indent=2)
        print(f"Mean best validation accuracy: {summary['mean_best_validation_accuracy']:.3f}%")
        print(f"Validation-only outputs: {output_dir}")
        return summary

    predictions = pd.DataFrame(prediction_rows)
    predictions.to_csv(output_dir / "predictions.csv", index=False)

    raw_cm = confusion_matrix(all_true, all_pred, labels=list(range(bundle.num_classes)))
    row_sum = raw_cm.sum(axis=1, keepdims=True)
    row_cm = np.divide(
        raw_cm.astype(np.float64), row_sum,
        out=np.zeros_like(raw_cm, dtype=np.float64), where=row_sum != 0
    )
    labels_index = bundle.class_names
    pd.DataFrame(raw_cm, index=labels_index, columns=labels_index).to_csv(
        output_dir / "confusion_matrix_raw.csv"
    )
    pd.DataFrame(row_cm, index=labels_index, columns=labels_index).to_csv(
        output_dir / "confusion_matrix_row_normalized.csv"
    )

    summary = {
        "dataset": dataset_key,
        "task": bundle.task,
        "class_names": bundle.class_names,
        "variant": args.variant,
        "seed": int(args.seed),
        "subjects": len(bundle.subjects),
        "mean_test_accuracy": float(results["test_accuracy"].mean()),
        "std_test_accuracy": float(results["test_accuracy"].std(ddof=0)),
        "mean_balanced_accuracy": float(results["balanced_accuracy"].mean()),
        "std_balanced_accuracy": float(results["balanced_accuracy"].std(ddof=0)),
        "mean_macro_f1": float(results["macro_f1"].mean()),
        "std_macro_f1": float(results["macro_f1"].std(ddof=0)),
        "pooled_accuracy_from_confusion_matrix": float(100.0 * np.trace(raw_cm) / max(raw_cm.sum(), 1)),
        "pooled_balanced_accuracy_from_confusion_matrix": float(100.0 * np.mean(np.diag(row_cm))),
        "mean_best_validation_accuracy": float(results["best_val_accuracy"].mean()),
        "confusion_matrix_raw": raw_cm.tolist(),
        "confusion_matrix_row_normalized": row_cm.tolist(),
        "interference_ratio": float(args.interference_ratio),
        "fuzzynum": int(args.fuzzynum),
        "curvature": float(args.curvature),
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    if electrode_rows:
        _save_electrode_outputs(output_dir, electrode_rows)

    print("\nLOSO summary")
    for key in ("mean_test_accuracy", "mean_balanced_accuracy", "mean_macro_f1"):
        print(f"{key}: {summary[key]:.3f}")
    print(f"Raw confusion matrix saved to: {output_dir / 'confusion_matrix_raw.csv'}")
    print(f"Outputs: {output_dir}")
    return summary
