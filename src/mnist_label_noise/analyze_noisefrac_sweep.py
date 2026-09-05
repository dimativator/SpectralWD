"""Aggregate the scaled-up (14.2M-param MLP) noise_frac x spectral_l1_reg_coef
sweep (scripts/mnist_label_noise/train_noisefrac_sweep.sh) into a summary
table and a plot mirroring the toy label-noise noisefrac-scan figure.

Usage: python src/mnist_label_noise/analyze_noisefrac_sweep.py
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parents[2] / "exps" / "mnist_label_noise"
OUT_PNG = Path(__file__).resolve().parents[2] / "spectral-wd-gen-scaled-noisefrac-scan.png"


def load_results() -> pd.DataFrame:
    rows = []
    for run_dir in sorted(RESULTS_DIR.iterdir()):
        result_path = run_dir / "result.json"
        if not result_path.exists():
            continue
        with open(result_path) as f:
            r = json.load(f)
        rows.append({
            "noise_frac": r["noise_frac"],
            "coef": r["spectral_l1_reg_coef"],
            "test_acc": r["final_test_acc"],
            "clean_train_acc": r["clean_label_train_acc"],
            "corrupted_train_acc": r["corrupted_label_train_acc"],
            "mean_erank": r["effective_ranks"]["effective_rank/mean_weighted"],
        })
    return pd.DataFrame(rows).sort_values(["noise_frac", "coef"]).reset_index(drop=True)


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for nf, g in df.groupby("noise_frac"):
        baseline = g[g["coef"] == 0].iloc[0]
        best = g.loc[g["test_acc"].idxmax()]
        rows.append({
            "noise_frac": nf,
            "baseline_test_acc": baseline["test_acc"],
            "best_test_acc": best["test_acc"],
            "best_coef": best["coef"],
            "gain_pp": (best["test_acc"] - baseline["test_acc"]) * 100,
        })
    return pd.DataFrame(rows)


def plot(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for nf, g in df.groupby("noise_frac"):
        axes[0].plot(g["coef"], g["test_acc"], marker="o", label=f"noise_frac={nf}")
        axes[1].plot(g["coef"], g["mean_erank"], marker="o", label=f"noise_frac={nf}")
    axes[0].set_xlabel("spectral_l1_reg_coef")
    axes[0].set_ylabel("final test acc (clean)")
    axes[0].set_title("Scaled MLP (14.2M params): test acc vs coef")
    axes[0].legend()
    axes[1].set_xlabel("spectral_l1_reg_coef")
    axes[1].set_ylabel("mean effective rank")
    axes[1].set_title("Mean effective rank vs coef")
    axes[1].set_yscale("log")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)
    print(f"saved {OUT_PNG}")


if __name__ == "__main__":
    df = load_results()
    print(df.to_string(index=False))
    summary = summary_table(df)
    print()
    print(summary.to_string(index=False))
    plot(df)
