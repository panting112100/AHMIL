"""Subject-level paired tests after averaging each subject's scores across seeds."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t, ttest_rel

METRICS = ["test_accuracy", "balanced_accuracy", "macro_f1"]


def load_method(root: Path, metric: str) -> pd.Series:
    files = sorted(root.glob("**/subject_results.csv"))
    if not files:
        raise FileNotFoundError(f"No subject_results.csv found under {root}")
    frames = []
    for file in files:
        frame = pd.read_csv(file)
        if "subject" not in frame or metric not in frame:
            raise ValueError(f"{file} must contain 'subject' and '{metric}' columns")
        frames.append(frame[["subject", metric]])
    combined = pd.concat(frames, ignore_index=True)
    return combined.groupby("subject")[metric].mean().sort_index()


def holm_adjust(p_values):
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values)
    running_max = 0.0
    m = len(p_values)
    for rank, index in enumerate(order):
        value = min(1.0, (m - rank) * p_values[index])
        running_max = max(running_max, value)
        adjusted[index] = running_max
    return adjusted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method_a", required=True, type=Path)
    parser.add_argument("--method_b", required=True, type=Path)
    parser.add_argument("--metrics", nargs="+", choices=METRICS, default=METRICS)
    parser.add_argument("--output", default="./paired_statistics.csv")
    args = parser.parse_args()

    rows = []
    raw_p = []
    for metric in args.metrics:
        a = load_method(args.method_a, metric)
        b = load_method(args.method_b, metric)
        common = sorted(set(a.index) & set(b.index))
        if len(common) < 2:
            raise ValueError(f"At least two common subjects are required for {metric}.")
        a_values = a.loc[common].to_numpy(dtype=float)
        b_values = b.loc[common].to_numpy(dtype=float)
        differences = a_values - b_values
        n = len(differences)
        mean_diff = float(np.mean(differences))
        sem = float(np.std(differences, ddof=1) / np.sqrt(n))
        critical = float(t.ppf(0.975, n - 1))
        if np.allclose(differences, differences[0]):
            if np.isclose(mean_diff, 0.0):
                t_statistic, p_value = 0.0, 1.0
            else:
                t_statistic, p_value = float(np.sign(mean_diff) * np.inf), 0.0
        else:
            test = ttest_rel(a_values, b_values, alternative="two-sided")
            t_statistic, p_value = float(test.statistic), float(test.pvalue)
        rows.append({
            "metric": metric,
            "n_subjects": n,
            "mean_paired_difference_A_minus_B": mean_diff,
            "ci95_lower": mean_diff - critical * sem,
            "ci95_upper": mean_diff + critical * sem,
            "paired_t": t_statistic,
            "p_value": p_value,
        })
        raw_p.append(p_value)

    adjusted = holm_adjust(raw_p)
    for row, p_adjusted in zip(rows, adjusted):
        row["holm_adjusted_p"] = float(p_adjusted)
    result = pd.DataFrame(rows)
    result.to_csv(args.output, index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
