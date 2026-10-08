import argparse
import csv
import dataclasses
import json
import logging
import math
from contextlib import nullcontext
from pathlib import Path

import torch

from optim.adamw_spectral_L1_reg import AdamWSpectralL1Reg

from .config import BertDeltaLabelNoiseConfig
from .data import get_data
from .model import (
    build_model,
    regularized_displacement_parameters,
    trainable_state_dict,
)

logger = logging.getLogger(__name__)


def _autocast(cfg: BertDeltaLabelNoiseConfig):
    if cfg.bf16 and cfg.device.startswith("cuda"):
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _model_inputs(
    batch: dict[str, torch.Tensor], device: str
) -> dict[str, torch.Tensor]:
    return {
        name: value.to(device, non_blocking=True)
        for name, value in batch.items()
        if name not in {"clean_label", "noisy_label", "corruption_mask"}
    }


def _classification_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    losses: float,
    num_labels: int,
) -> dict[str, float]:
    accuracy = (predictions == targets).float().mean().item()
    f1_values = []
    for label in range(num_labels):
        predicted = predictions == label
        actual = targets == label
        true_positive = (predicted & actual).sum().item()
        false_positive = (predicted & ~actual).sum().item()
        false_negative = (~predicted & actual).sum().item()
        denominator = 2 * true_positive + false_positive + false_negative
        f1_values.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
    return {
        "loss": losses,
        "accuracy": accuracy,
        "macro_f1": sum(f1_values) / len(f1_values),
    }


@torch.no_grad()
def evaluate(model, loader, cfg: BertDeltaLabelNoiseConfig) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    count = 0
    predictions = []
    targets = []
    for batch in loader:
        labels = batch["clean_label"].to(cfg.device, non_blocking=True)
        with _autocast(cfg):
            output = model(**_model_inputs(batch, cfg.device), labels=labels)
        total_loss += output.loss.item() * len(labels)
        count += len(labels)
        predictions.append(output.logits.argmax(dim=-1).cpu())
        targets.append(labels.cpu())
    return _classification_metrics(
        torch.cat(predictions),
        torch.cat(targets),
        total_loss / count,
        cfg.num_labels,
    )


@torch.no_grad()
def train_partition_metrics(
    model, loader, cfg: BertDeltaLabelNoiseConfig
) -> dict[str, float | None]:
    model.eval()
    clean_correct = 0
    clean_count = 0
    corrupted_observed_correct = 0
    corrupted_original_correct = 0
    corrupted_count = 0
    for batch in loader:
        clean_labels = batch["clean_label"].to(cfg.device, non_blocking=True)
        noisy_labels = batch["noisy_label"].to(cfg.device, non_blocking=True)
        corruption_mask = batch["corruption_mask"].to(cfg.device, non_blocking=True)
        with _autocast(cfg):
            logits = model(**_model_inputs(batch, cfg.device)).logits
        predictions = logits.argmax(dim=-1)
        clean_mask = ~corruption_mask
        clean_correct += (
            (predictions[clean_mask] == clean_labels[clean_mask]).sum().item()
        )
        clean_count += clean_mask.sum().item()
        corrupted_observed_correct += (
            (predictions[corruption_mask] == noisy_labels[corruption_mask]).sum().item()
        )
        corrupted_original_correct += (
            (predictions[corruption_mask] == clean_labels[corruption_mask]).sum().item()
        )
        corrupted_count += corruption_mask.sum().item()
    return {
        "clean_subset_fit": clean_correct / max(clean_count, 1),
        "corrupted_label_fit": (
            corrupted_observed_correct / corrupted_count if corrupted_count else None
        ),
        "original_label_recovery": (
            corrupted_original_correct / corrupted_count if corrupted_count else None
        ),
    }


@torch.no_grad()
def displacement_ranks(
    model,
    references: dict[str, torch.Tensor],
    regularized_names: list[str],
) -> dict[str, float]:
    parameters = dict(model.named_parameters())
    output: dict[str, float] = {}
    weighted_effective = 0.0
    total_weight = 0
    for name in regularized_names:
        parameter = parameters[name].detach().float()
        delta = parameter - references[name].to(parameter.device, dtype=parameter.dtype)
        singular_values = torch.linalg.svdvals(delta)
        total = singular_values.sum().item()
        if total <= 1e-12:
            effective_rank = 0.0
            stable_rank = 0.0
        else:
            probabilities = singular_values / singular_values.sum()
            effective_rank = torch.exp(
                -(probabilities * probabilities.clamp_min(1e-12).log()).sum()
            ).item()
            stable_rank = (
                delta.square().sum() / singular_values[0].square().clamp_min(1e-12)
            ).item()
        output[f"effective_rank_delta/{name}"] = effective_rank
        output[f"stable_rank_delta/{name}"] = stable_rank
        weight = min(delta.shape)
        weighted_effective += effective_rank * weight
        total_weight += weight
    output["effective_rank_delta/mean_weighted"] = weighted_effective / total_weight
    return output


def build_optimizer(model, cfg: BertDeltaLabelNoiseConfig):
    names, regularized = regularized_displacement_parameters(
        model,
        include_classifier=cfg.regularize_classifier,
    )
    regularized_ids = {id(parameter) for parameter in regularized}
    unregularized = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in regularized_ids
    ]
    groups = [
        {
            "params": regularized,
            "spectral_l1_reg_coef": cfg.spectral_delta_coef,
            "matrix_l2_reg_coef": cfg.matrix_l2_delta_coef,
            "regularize_from_init": True,
            "weight_decay": 0.0,
        },
        {
            "params": unregularized,
            "spectral_l1_reg_coef": 0.0,
            "matrix_l2_reg_coef": 0.0,
            "regularize_from_init": False,
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamWSpectralL1Reg(
        groups,
        lr=cfg.lr,
        betas=(cfg.beta1, cfg.beta2),
        spectral_l1_reg_coef=0.0,
        matrix_l2_reg_coef=0.0,
        spectral_wd_order=cfg.spectral_wd_order,
        l2_wd_order=cfg.l2_wd_order,
        weight_decay=0.0,
        regularize_from_init=False,
    )
    return optimizer, names


def train_one_run(cfg: BertDeltaLabelNoiseConfig) -> dict:
    if cfg.spectral_delta_coef > 0 and cfg.matrix_l2_delta_coef > 0:
        raise ValueError("Spectral-delta and matrix L2-delta are mutually exclusive")
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        # Explicit initialization avoids a lazy-CUDA reset failure when a
        # physical GPU is remapped to cuda:0 through CUDA_VISIBLE_DEVICES.
        torch.cuda.set_device(cfg.device)
        torch.cuda.manual_seed_all(cfg.seed)
        torch.cuda.reset_peak_memory_stats(cfg.device)
    if cfg.spectral_delta_coef > 0:
        family = "spectral-delta"
        coefficient = cfg.spectral_delta_coef
    elif cfg.matrix_l2_delta_coef > 0:
        family = "l2-sp"
        coefficient = cfg.matrix_l2_delta_coef
    else:
        family = "no-reg"
        coefficient = 0.0
    run_name = cfg.run_name or (
        f"bert_{cfg.dataset}_nf{cfg.noise_frac}_{family}_{coefficient}_s{cfg.seed}"
    )
    if cfg.run_name is None:
        order = cfg.l2_wd_order if family == "l2-sp" else cfg.spectral_wd_order
        run_name += f"_wd{order}"
    result_path = Path(cfg.results_base_folder) / run_name / "result.json"
    if result_path.exists():
        logger.info("skip completed %s", run_name)
        return json.loads(result_path.read_text())

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name, cache_dir=cfg.datasets_dir
    )
    data = get_data(cfg, tokenizer)
    # Re-seed after data construction so model initialization is invariant to cache state.
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    model = build_model(cfg).to(cfg.device)
    optimizer, regularized_names = build_optimizer(model, cfg)
    named_parameters = dict(model.named_parameters())
    references = {
        name: named_parameters[name].detach().cpu().clone()
        for name in regularized_names
    }
    total_steps = cfg.epochs * len(data.train_loader)
    warmup_steps = max(1, round(cfg.warmup_fraction * total_steps))

    def lr_multiplier(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_multiplier)
    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    logger.info(
        "run=%s family=%s coefficient=%g params=%d trainable=%d regularized_matrices=%d "
        "train=%d val=%d test=%d split_hash=%s",
        run_name,
        family,
        coefficient,
        total_params,
        trainable_params,
        len(regularized_names),
        data.train_size,
        data.validation_size,
        data.test_size,
        data.split_hash,
    )

    history = []
    peak_validation_accuracy = -math.inf
    peak_validation_epoch = 0
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_train_loss = 0.0
        train_count = 0
        for batch in data.train_loader:
            labels = batch["noisy_label"].to(cfg.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with _autocast(cfg):
                output = model(**_model_inputs(batch, cfg.device), labels=labels)
                loss = output.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [
                    parameter
                    for parameter in model.parameters()
                    if parameter.requires_grad
                ],
                cfg.grad_clip,
            )
            optimizer.step()
            scheduler.step()
            total_train_loss += loss.item() * len(labels)
            train_count += len(labels)

        validation = evaluate(model, data.validation_loader, cfg)
        partitions = {
            "clean_subset_fit": None,
            "corrupted_label_fit": None,
            "original_label_recovery": None,
        }
        if (
            epoch == 1
            or epoch % cfg.partition_eval_interval == 0
            or epoch == cfg.epochs
        ):
            partitions = train_partition_metrics(model, data.train_eval_loader, cfg)
        row = {
            "epoch": epoch,
            "train_noisy_loss": total_train_loss / train_count,
            "validation_loss": validation["loss"],
            "validation_accuracy": validation["accuracy"],
            "validation_macro_f1": validation["macro_f1"],
            **partitions,
        }
        history.append(row)
        if row["validation_accuracy"] > peak_validation_accuracy:
            peak_validation_accuracy = row["validation_accuracy"]
            peak_validation_epoch = epoch
        logger.info(
            "run=%s epoch=%d/%d train_loss=%.4f val_loss=%.4f val_acc=%.4f "
            "val_f1=%.4f clean_fit=%s corrupt_fit=%s recovery=%s",
            run_name,
            epoch,
            cfg.epochs,
            row["train_noisy_loss"],
            row["validation_loss"],
            row["validation_accuracy"],
            row["validation_macro_f1"],
            row["clean_subset_fit"],
            row["corrupted_label_fit"],
            row["original_label_recovery"],
        )

    final_partitions = train_partition_metrics(model, data.train_eval_loader, cfg)
    final_state = trainable_state_dict(model)
    final_test = evaluate(model, data.test_loader, cfg) if cfg.evaluate_test else None
    final_ranks = (
        {}
        if cfg.skip_rank_analysis
        else displacement_ranks(model, references, regularized_names)
    )
    result = {
        "run_name": run_name,
        "family": family,
        "coefficient": coefficient,
        "seed": cfg.seed,
        "split_seed": cfg.split_seed,
        "data_seed": cfg.data_seed,
        "noise_frac": cfg.noise_frac,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "regularized_matrices": regularized_names,
        "split_hash": data.split_hash,
        "selection_policy": "fixed_training_horizon",
        "selected_epoch": cfg.epochs,
        "peak_validation_epoch": peak_validation_epoch,
        "peak_validation_accuracy": peak_validation_accuracy,
        "selected_validation": {
            "loss": history[-1]["validation_loss"],
            "accuracy": history[-1]["validation_accuracy"],
            "macro_f1": history[-1]["validation_macro_f1"],
        },
        "validation_selected_test": final_test,
        "selected_train_partitions": final_partitions,
        "final_train_partitions": final_partitions,
        "final_validation": {
            "loss": history[-1]["validation_loss"],
            "accuracy": history[-1]["validation_accuracy"],
            "macro_f1": history[-1]["validation_macro_f1"],
        },
        "final_test": final_test,
        "final_displacement_ranks": final_ranks,
        "displacement_ranks": final_ranks,
        "history": history,
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated(cfg.device))
            if torch.cuda.is_available()
            else None
        ),
        "config": dataclasses.asdict(cfg),
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    if cfg.save_final_checkpoint:
        torch.save(
            {"epoch": cfg.epochs, "trainable_state_dict": final_state},
            result_path.parent / "final_trainable_checkpoint.pt",
        )
    result_path.write_text(json.dumps(result, indent=2))
    with (result_path.parent / "history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    logger.info("saved results to %s", result_path.parent)
    if result["peak_cuda_memory_bytes"] is not None:
        logger.info(
            "peak_cuda_memory=%.2f GiB",
            result["peak_cuda_memory_bytes"] / (1024**3),
        )
    return result


def parse_args() -> BertDeltaLabelNoiseConfig:
    parser = argparse.ArgumentParser()
    defaults = dataclasses.asdict(BertDeltaLabelNoiseConfig())
    for field in dataclasses.fields(BertDeltaLabelNoiseConfig):
        default = defaults[field.name]
        if isinstance(default, bool):
            parser.add_argument(
                f"--{field.name}",
                action=argparse.BooleanOptionalAction,
                default=default,
            )
        else:
            parser.add_argument(
                f"--{field.name}",
                type=type(default) if default is not None else str,
                default=default,
            )
    return BertDeltaLabelNoiseConfig(**vars(parser.parse_args()))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    torch.set_float32_matmul_precision("high")
    train_one_run(parse_args())
