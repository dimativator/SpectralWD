"""Launch the paper's 124M, 257M, or 500M LLaMA Adam baselines."""

import argparse
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from .regularization import Regularization


@dataclass(frozen=True)
class LlamaProfile:
    width: int
    heads: int
    layers: int
    iterations: int
    batch_size: int
    accumulation_steps: int
    learning_rate: float
    eval_interval: int
    checkpoint_interval: int


PROFILES = {
    "124m": LlamaProfile(768, 12, 12, 19000, 64, 2, 5e-3, 137, 1000),
    "257m": LlamaProfile(1024, 16, 16, 39000, 32, 4, 3e-3, 281, 2000),
    "500m": LlamaProfile(1280, 20, 22, 76294, 16, 8, 1e-3, 281, 2000),
}


def build_command(args: argparse.Namespace) -> list[str]:
    profile = PROFILES[args.size]
    regularization = Regularization(args.method, args.coefficient)
    optimizer = "adamw-spectral-l1-reg" if args.method == "spectral" else "adamw"
    weight_decay = (
        args.spectral_nonmatrix_weight_decay
        if args.method == "spectral"
        else regularization.l2_coefficient
    )
    run_name = args.run_name or (
        f"llama{args.size}_{args.method}_coef{args.coefficient:g}_"
        f"lr{profile.learning_rate:g}_finewebedu"
    )
    command = [
        args.python,
        "./src/main.py",
        "--experiment_name",
        run_name,
        "--results_base_folder",
        str(args.results_base_folder),
        "--model",
        "llama",
        "--datasets_dir",
        str(args.datasets_dir),
        "--dataset",
        "finewebedu",
        "--tokenized_data_dir",
        str(args.tokenized_data_dir),
        "--opt",
        optimizer,
        "--lr",
        str(profile.learning_rate),
        "--iterations",
        str(profile.iterations),
        "--n_embd",
        str(profile.width),
        "--n_head",
        str(profile.heads),
        "--n_layer",
        str(profile.layers),
        "--batch_size",
        str(profile.batch_size),
        "--sequence_length",
        "1024",
        "--acc_steps",
        str(profile.accumulation_steps),
        "--grad_clip",
        "0.5",
        "--seed",
        str(args.seed),
        "--weight_decay",
        str(weight_decay),
        "--spectral_l1_reg_coef",
        str(regularization.spectral_coefficient),
        "--scheduler",
        "cos",
        "--warmup_steps",
        "2000",
        "--dropout",
        "0",
        "--beta1",
        "0.9",
        "--beta2",
        "0.95",
        "--eval_interval",
        str(profile.eval_interval),
        "--latest_ckpt_interval",
        str(profile.checkpoint_interval),
        "--log_interval",
        "4",
        "--finewebedu_max_files",
        str(args.finewebedu_max_files),
        "--effective_rank_interval",
        str(args.effective_rank_interval),
        "--device",
        args.device,
    ]
    if args.downstream_eval:
        command.extend(
            [
                "--downstream_eval_enabled",
                "--downstream_eval_interval",
                str(args.downstream_eval_interval),
                "--downstream_task_group",
                "basic_v2",
            ]
        )
    if args.wandb:
        command.extend(["--wandb", "--wandb_project", args.wandb_project])
        if args.wandb_entity is not None:
            command.extend(["--wandb_entity", args.wandb_entity])
    command.extend(args.extra_arg)
    return command


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", required=True, choices=tuple(PROFILES))
    parser.add_argument("--method", required=True, choices=("no_wd", "l2", "spectral"))
    parser.add_argument("--coefficient", type=float, default=0.0)
    parser.add_argument("--spectral-nonmatrix-weight-decay", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--datasets-dir", type=Path, default=Path("datasets"))
    parser.add_argument(
        "--tokenized-data-dir",
        type=Path,
        default=Path("data/fineweb-edu-tokenized"),
    )
    parser.add_argument(
        "--results-base-folder", type=Path, default=Path("exps/paper_llama")
    )
    parser.add_argument("--run-name")
    parser.add_argument("--finewebedu-max-files", type=int, default=5)
    parser.add_argument("--effective-rank-interval", type=int, default=500)
    parser.add_argument("--downstream-eval", action="store_true")
    parser.add_argument("--downstream-eval-interval", type=int, default=4000)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="spectral-wd")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--extra-arg", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    command = build_command(args)
    print(shlex.join(command))
    if not args.dry_run:
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
