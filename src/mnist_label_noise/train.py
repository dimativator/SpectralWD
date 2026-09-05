"""Label-noise generalization study (scaled-up model), integrated into the
main repo. Reuses AdamWSpectralL1Reg (optim/adamw_spectral_L1_reg.py) and the
effective-rank / truncated-SVD utilities (models/compress.py) unmodified.

Must be run with `src/` on sys.path, e.g. from the repo root:
    PYTHONPATH=./src python -m mnist_label_noise.train --noise_frac 0.25 --spectral_l1_reg_coef 1.0
See scripts/mnist_label_noise/train_noisefrac_sweep.sh for the full sweep.
"""

import argparse
import copy
import csv
import dataclasses
import json
import logging
from pathlib import Path

import torch
import torch.nn.functional as F

from models.compress import compress_model_svd, model_effective_ranks
from optim.adamw_spectral_L1_reg import AdamWSpectralL1Reg

from .config import MLPLabelNoiseConfig
from .data import get_loaders
from .model import get_model

logger = logging.getLogger(__name__)

RANK_GRID = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += F.cross_entropy(logits, y, reduction="sum").item()
        correct += (logits.argmax(dim=-1) == y).sum().item()
        n += y.size(0)
    model.train()
    return total_loss / n, correct / n


@torch.no_grad()
def clean_vs_noisy_train_acc(model, train_subset, device):
    model.eval()
    loader = torch.utils.data.DataLoader(train_subset, batch_size=1024, shuffle=False)
    correct_clean, n_clean, correct_noisy, n_noisy = 0, 0, 0, 0
    offset = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(dim=-1)
        batch_is_noisy = train_subset.is_noisy[offset: offset + y.size(0)]
        offset += y.size(0)
        correct = pred == y
        correct_clean += correct[~batch_is_noisy].sum().item()
        n_clean += (~batch_is_noisy).sum().item()
        correct_noisy += correct[batch_is_noisy].sum().item()
        n_noisy += batch_is_noisy.sum().item()
    model.train()
    return correct_clean / max(n_clean, 1), correct_noisy / max(n_noisy, 1)


@torch.no_grad()
def truncation_sweep(model, test_loader, device):
    curve = []
    for rank in RANK_GRID:
        compressed = compress_model_svd(copy.deepcopy(model), rank=rank, skip_names=()).to(device)
        _, acc = evaluate(compressed, test_loader, device)
        curve.append({"rank": rank, "test_acc": acc})
    return curve


def train_one_run(cfg: MLPLabelNoiseConfig) -> dict:
    torch.manual_seed(cfg.seed)
    if cfg.spectral_l1_reg_coef > 0 and cfg.matrix_l2_reg_coef > 0:
        raise ValueError("Spectral and matrix L2 regularization are mutually exclusive")
    if cfg.matrix_l2_reg_coef > 0:
        family, coefficient = "l2", cfg.matrix_l2_reg_coef
    elif cfg.spectral_l1_reg_coef > 0:
        family, coefficient = "spectral", cfg.spectral_l1_reg_coef
    else:
        family, coefficient = "no_wd", 0.0
    regularizer_tag = "no_wd" if family == "no_wd" else f"{family}wd"
    run_name = cfg.run_name or (
        f"mlp_h{cfg.hidden_dim}x{cfg.n_hidden_layers}_nf{cfg.noise_frac}_"
        f"{regularizer_tag}_{coefficient}_seed{cfg.seed}"
    )
    result_path = Path(cfg.results_base_folder) / run_name / "result.json"
    if result_path.exists():
        logger.info("skip completed %s", run_name)
        return json.loads(result_path.read_text())

    if cfg.wandb:
        import wandb
        wandb.init(project=cfg.wandb_project, entity=cfg.wandb_entity, name=run_name, config=dataclasses.asdict(cfg))

    train_loader, train_eval_loader, val_loader, test_loader, train_subset = get_loaders(cfg)

    model = get_model("mlp", cfg).to(cfg.device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("run=%s params=%.2fM", run_name, n_params / 1e6)

    optimizer = AdamWSpectralL1Reg(
        model.parameters(),
        lr=cfg.lr,
        betas=(cfg.beta1, cfg.beta2),
        spectral_l1_reg_coef=cfg.spectral_l1_reg_coef,
        matrix_l2_reg_coef=cfg.matrix_l2_reg_coef,
        svt_interval=0,
    )

    history = []
    for epoch in range(cfg.epochs):
        for x, y in train_loader:
            x, y = x.to(cfg.device), y.to(cfg.device)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()

        train_loss, train_acc = evaluate(model, train_eval_loader, cfg.device)
        val_loss, val_acc = (
            evaluate(model, val_loader, cfg.device)
            if val_loader is not None
            else (None, None)
        )
        should_evaluate_test = cfg.evaluate_test_each_epoch or epoch == cfg.epochs - 1
        test_loss, test_acc = (
            evaluate(model, test_loader, cfg.device)
            if should_evaluate_test
            else (None, None)
        )
        clean_acc_ep, noisy_acc_ep = clean_vs_noisy_train_acc(model, train_subset, cfg.device)
        row = {
            "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
            "test_loss": test_loss, "test_acc": test_acc,
            "validation_loss": val_loss, "validation_acc": val_acc,
            "clean_label_train_acc": clean_acc_ep, "corrupted_label_train_acc": noisy_acc_ep,
        }
        history.append(row)
        if cfg.wandb:
            wandb.log(row, step=epoch)
        if epoch % 10 == 0 or epoch == cfg.epochs - 1:
            logger.info(
                "run=%s epoch=%d train_acc=%.4f val_acc=%s test_acc=%.4f corrupted_train_acc=%.4f",
                run_name, epoch, train_acc,
                f"{val_acc:.4f}" if val_acc is not None else "n/a",
                test_acc if test_acc is not None else float("nan"), noisy_acc_ep,
            )

    # MPS does not implement torch.linalg.svd, while CUDA does and avoids a
    # severe shared-CPU bottleneck for the 2048x2048 matrices in this model.
    rank_device = cfg.device if str(cfg.device).startswith("cuda") else "cpu"
    rank_model = copy.deepcopy(model).to(rank_device)
    ranks = (
        {} if cfg.skip_rank_analysis else model_effective_ranks(rank_model, skip_names=())
    )
    curve = (
        []
        if cfg.skip_rank_analysis or cfg.skip_truncation_sweep
        else truncation_sweep(rank_model.to("cpu"), test_loader, "cpu")
    )
    final = history[-1]
    result = {
        "run_name": run_name,
        "seed": cfg.seed,
        "spectral_l1_reg_coef": cfg.spectral_l1_reg_coef,
        "matrix_l2_reg_coef": cfg.matrix_l2_reg_coef,
        "regularizer": family,
        "noise_frac": cfg.noise_frac,
        "selection_policy": "fixed_training_horizon",
        "selected_epoch": cfg.epochs,
        "n_params": n_params,
        "final_train_acc": final["train_acc"],
        "final_test_acc": final["test_acc"],
        "final_validation_acc": final["validation_acc"],
        "clean_label_train_acc": final["clean_label_train_acc"],
        "corrupted_label_train_acc": final["corrupted_label_train_acc"],
        "effective_ranks": ranks,
        "history": history,
        "truncation_curve": curve,
        "config": dataclasses.asdict(cfg),
    }

    if cfg.wandb:
        import wandb
        wandb.log({"effective_ranks": ranks, "truncation_curve": curve})
        wandb.finish()

    run_dir = result_path.parent
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    with open(run_dir / "history.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    logger.info("saved results to %s", run_dir)

    return result


def parse_args() -> MLPLabelNoiseConfig:
    p = argparse.ArgumentParser()
    defaults = dataclasses.asdict(MLPLabelNoiseConfig())
    for field in dataclasses.fields(MLPLabelNoiseConfig):
        default = defaults[field.name]
        if field.type == bool:
            p.add_argument(f"--{field.name}", action="store_true", default=default)
        else:
            p.add_argument(f"--{field.name}", type=type(default) if default is not None else str, default=default)
    args = p.parse_args()
    return MLPLabelNoiseConfig(**vars(args))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cfg = parse_args()
    train_one_run(cfg)
