import argparse
import csv
import dataclasses
import json
import logging
from pathlib import Path

import torch
import torch.nn.functional as F

from models.compress import effective_rank, stable_rank
from optim.adamw_spectral_L1_reg import AdamWSpectralL1Reg

from .config import GRULabelNoiseConfig
from .data import get_loaders
from .model import RowSequentialGRU

logger = logging.getLogger(__name__)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss, correct, count = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        total_loss += F.cross_entropy(logits, labels, reduction="sum").item()
        correct += (logits.argmax(dim=-1) == labels).sum().item()
        count += labels.numel()
    model.train()
    return total_loss / count, correct / count


@torch.no_grad()
def train_partition_metrics(model, train_subset, device):
    model.eval()
    loader = torch.utils.data.DataLoader(train_subset, batch_size=1024, shuffle=False)
    clean_correct = corrupted_observed_correct = corrupted_original_correct = 0
    clean_count = corrupted_count = offset = 0
    for images, observed_labels in loader:
        batch_size = observed_labels.numel()
        noisy_mask = train_subset.is_noisy[offset: offset + batch_size].to(device)
        original_labels = train_subset.original_labels[
            offset: offset + batch_size
        ].to(device)
        offset += batch_size
        observed_labels = observed_labels.to(device)
        predictions = model(images.to(device)).argmax(dim=-1)
        clean_correct += (predictions[~noisy_mask] == original_labels[~noisy_mask]).sum().item()
        corrupted_observed_correct += (
            predictions[noisy_mask] == observed_labels[noisy_mask]
        ).sum().item()
        corrupted_original_correct += (
            predictions[noisy_mask] == original_labels[noisy_mask]
        ).sum().item()
        clean_count += (~noisy_mask).sum().item()
        corrupted_count += noisy_mask.sum().item()
    model.train()
    return {
        "clean_subset_train_acc": clean_correct / max(clean_count, 1),
        "corrupted_label_train_acc": corrupted_observed_correct / max(corrupted_count, 1),
        "corrupted_original_train_acc": corrupted_original_correct / max(corrupted_count, 1),
    }


@torch.no_grad()
def matrix_ranks(model):
    ranks = {}
    weighted_effective = 0.0
    weighted_stable = 0.0
    total_weight = 0
    for name, parameter in model.named_parameters():
        if parameter.ndim != 2:
            continue
        matrix = parameter.detach()
        e_rank = effective_rank(matrix)
        s_rank = stable_rank(matrix)
        ranks[f"effective_rank/{name}"] = e_rank
        ranks[f"stable_rank/{name}"] = s_rank
        weight = min(matrix.shape)
        weighted_effective += e_rank * weight
        weighted_stable += s_rank * weight
        total_weight += weight
    ranks["effective_rank/mean_weighted"] = weighted_effective / total_weight
    ranks["stable_rank/mean_weighted"] = weighted_stable / total_weight
    return ranks


def train_one_run(cfg: GRULabelNoiseConfig):
    if cfg.spectral_l1_reg_coef > 0 and cfg.matrix_l2_reg_coef > 0:
        raise ValueError("Spectral and matrix L2 regularization are mutually exclusive")
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    if cfg.matrix_l2_reg_coef > 0:
        family, coefficient = "l2", cfg.matrix_l2_reg_coef
    elif cfg.spectral_l1_reg_coef > 0:
        family, coefficient = "spectral", cfg.spectral_l1_reg_coef
    else:
        family, coefficient = "no_wd", 0.0
    regularizer_tag = "no_wd" if family == "no_wd" else f"{family}wd"
    run_name = cfg.run_name or (
        f"gru_h{cfg.hidden_dim}x{cfg.n_layers}_rowseq_nf{cfg.noise_frac}_"
        f"{regularizer_tag}_{coefficient}_seed{cfg.seed}"
    )
    result_path = Path(cfg.results_base_folder) / run_name / "result.json"
    if result_path.exists():
        logger.info("skip completed %s", run_name)
        return json.loads(result_path.read_text())

    train_loader, train_eval_loader, val_loader, test_loader, train_subset = get_loaders(cfg)
    model = RowSequentialGRU(cfg).to(cfg.device)
    n_params = sum(parameter.numel() for parameter in model.parameters())
    matrix_names = [name for name, parameter in model.named_parameters() if parameter.ndim == 2]
    logger.info("run=%s params=%d matrices=%s", run_name, n_params, matrix_names)

    optimizer = AdamWSpectralL1Reg(
        model.parameters(),
        lr=cfg.lr,
        betas=(cfg.beta1, cfg.beta2),
        spectral_l1_reg_coef=cfg.spectral_l1_reg_coef,
        matrix_l2_reg_coef=cfg.matrix_l2_reg_coef,
        weight_decay=0.0,
        svt_interval=0,
    )

    history = []
    for epoch in range(cfg.epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(cfg.device), labels.to(cfg.device)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(images), labels)
            loss.backward()
            optimizer.step()

        train_loss, train_acc = evaluate(model, train_eval_loader, cfg.device)
        val_loss, val_acc = evaluate(model, val_loader, cfg.device)
        should_evaluate_test = cfg.evaluate_test_each_epoch or epoch + 1 == cfg.epochs
        test_loss, test_acc = (
            evaluate(model, test_loader, cfg.device)
            if should_evaluate_test
            else (None, None)
        )
        partitions = train_partition_metrics(model, train_subset, cfg.device)
        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "validation_loss": val_loss,
            "validation_acc": val_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
            **partitions,
        }
        history.append(row)
        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch + 1 == cfg.epochs:
            logger.info(
                "run=%s epoch=%d train=%.4f val=%.4f test=%.4f corrupt_fit=%.4f corrupt_recovery=%.4f",
                run_name,
                epoch + 1,
                train_acc,
                val_acc,
                test_acc if test_acc is not None else float("nan"),
                partitions["corrupted_label_train_acc"],
                partitions["corrupted_original_train_acc"],
            )

    ranks = {} if cfg.skip_rank_analysis else matrix_ranks(model)
    final = history[-1]
    result = {
        "run_name": run_name,
        "seed": cfg.seed,
        "regularizer": family,
        "spectral_l1_reg_coef": cfg.spectral_l1_reg_coef,
        "matrix_l2_reg_coef": cfg.matrix_l2_reg_coef,
        "noise_frac": cfg.noise_frac,
        "selection_policy": "fixed_training_horizon",
        "selected_epoch": cfg.epochs,
        "n_params": n_params,
        "regularized_matrices": matrix_names,
        "final_train_acc": final["train_acc"],
        "final_validation_acc": final["validation_acc"],
        "final_test_acc": final["test_acc"],
        "clean_subset_train_acc": final["clean_subset_train_acc"],
        "corrupted_label_train_acc": final["corrupted_label_train_acc"],
        "corrupted_original_train_acc": final["corrupted_original_train_acc"],
        "effective_ranks": ranks,
        "history": history,
        "config": dataclasses.asdict(cfg),
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2))
    with (result_path.parent / "history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    logger.info("saved results to %s", result_path.parent)
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    defaults = dataclasses.asdict(GRULabelNoiseConfig())
    for field in dataclasses.fields(GRULabelNoiseConfig):
        default = defaults[field.name]
        if field.type == bool:
            parser.add_argument(f"--{field.name}", action="store_true", default=default)
        else:
            parser.add_argument(
                f"--{field.name}",
                type=type(default) if default is not None else str,
                default=default,
            )
    return GRULabelNoiseConfig(**vars(parser.parse_args()))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    train_one_run(parse_args())
