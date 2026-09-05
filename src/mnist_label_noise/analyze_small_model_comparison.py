"""Compares hidden=2048 MLP + spectral WD against plain (unregularized) small
MLPs trained directly at a range of hidden_dim, to test whether spectral WD's
rank restriction is equivalent to just training a smaller model. The small
model grid spans hidden_dim=8..1448 (log-spaced) plus 5 exact targets matched
to the effective ranks the large model reaches at coef in {0.5,1,2,3,5} -- see
scripts/mnist_label_noise/train_small_model_comparison_sweep.sh (matched
targets) and train_small_model_gridscan_sweep.sh (the filling grid) for the
run scripts.

Usage: python src/mnist_label_noise/analyze_small_model_comparison.py
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SMALL_DIR = REPO_ROOT / "exps" / "mnist_small_model_comparison"
LARGE_DIR = REPO_ROOT / "exps" / "mnist_label_noise"
OUT_PNG = REPO_ROOT / "spectral-wd-vs-small-model-comparison.png"

LARGE_COEFS = [0, 0.5, 1, 2, 3, 5]
# exact matched targets (hidden_dim, matched coef) from train_small_model_comparison_sweep.sh
MATCHED_TARGETS = {30: 5, 73: 3, 135: 2, 329: 1, 745: 0.5}


def load_small_models() -> pd.DataFrame:
    rows = []
    for run_dir in sorted(SMALL_DIR.iterdir()):
        result_path = run_dir / "result.json"
        if not result_path.exists():
            continue
        with open(result_path) as f:
            r = json.load(f)
        hd = int(run_dir.name.split("_h")[1].split("x")[0])
        rows.append({
            "hidden_dim": hd,
            "erank": r["effective_ranks"]["effective_rank/mean_weighted"],
            "test_acc": r["final_test_acc"],
            "corrupted_train_acc": r["corrupted_label_train_acc"],
            "n_params": r["n_params"],
            "is_matched_target": hd in MATCHED_TARGETS,
        })
    return pd.DataFrame(rows).sort_values("hidden_dim").reset_index(drop=True)


def load_large_models() -> pd.DataFrame:
    rows = []
    for coef in LARGE_COEFS:
        r = json.load(open(LARGE_DIR / f"mlp_h2048x4_nf0.25_sl1_{coef}_seed0" / "result.json"))
        rows.append({
            "coef": coef,
            "erank": r["effective_ranks"]["effective_rank/mean_weighted"],
            "test_acc": r["final_test_acc"],
            "corrupted_train_acc": r["corrupted_label_train_acc"],
            "n_params": r["n_params"],
        })
    return pd.DataFrame(rows).sort_values("erank").reset_index(drop=True)


def matched_pairs_table(small: pd.DataFrame, large: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for hd, coef in MATCHED_TARGETS.items():
        s = small[small["hidden_dim"] == hd].iloc[0]
        l = large[large["coef"] == coef].iloc[0]
        rows.append({
            "target_hidden_dim": hd,
            "matched_coef": coef,
            "small_erank": s["erank"], "small_test_acc": s["test_acc"],
            "small_corrupted_acc": s["corrupted_train_acc"], "small_n_params": s["n_params"],
            "large_erank": l["erank"], "large_test_acc": l["test_acc"],
            "large_corrupted_acc": l["corrupted_train_acc"], "large_n_params": l["n_params"],
        })
    return pd.DataFrame(rows).sort_values("target_hidden_dim")


def plot(small: pd.DataFrame, large: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    for ax, metric, title in [
        (axes[0], "test_acc", "Test acc: small-model curve vs spectral WD"),
        (axes[1], "corrupted_train_acc", "Noise memorization: small-model curve vs spectral WD"),
    ]:
        ax.plot(small["erank"], small[metric], marker=".", color="tab:blue", alpha=0.6,
                label="plain small model (coef=0), full grid")
        matched = small[small["is_matched_target"]]
        ax.scatter(matched["erank"], matched[metric], marker="o", s=70, color="tab:blue",
                   edgecolor="black", zorder=5, label="plain small model, rank-matched target")
        ax.plot(large["erank"], large[metric], marker="s", color="tab:red",
                label="hidden=2048 + spectral WD")
        ax.set_xscale("log")
        ax.set_xlabel("effective rank (mean_weighted, achieved)")
        ax.set_title(title)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("final test acc")
    axes[1].set_ylabel("corrupted-label train acc")

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)
    print(f"saved {OUT_PNG}")


if __name__ == "__main__":
    small = load_small_models()
    large = load_large_models()

    print("=== full small-model grid ===")
    print(small.to_string(index=False))
    print("\n=== large model (hidden=2048) + spectral WD ===")
    print(large.to_string(index=False))
    print("\n=== matched-target pairs ===")
    print(matched_pairs_table(small, large).to_string(index=False))

    plot(small, large)
