"""Run one fixed-horizon robustness experiment from the paper."""

import argparse
import dataclasses
import json
from pathlib import Path

from .regularization import Regularization


BERT_DATASETS = {
    "ag_news": {
        "num_labels": 4,
        "train_size": 5000,
        "validation_size": 5000,
        "test_size": 7600,
    },
    "dbpedia_14": {
        "num_labels": 14,
        "train_size": 4200,
        "validation_size": 4200,
        "test_size": 9800,
    },
    "yahoo_answers_topics": {
        "num_labels": 10,
        "train_size": 5000,
        "validation_size": 5000,
        "test_size": 10000,
    },
    "yelp_review_full": {
        "num_labels": 5,
        "train_size": 5000,
        "validation_size": 5000,
        "test_size": 10000,
    },
}


def build_config(args: argparse.Namespace):
    regularization = Regularization(args.method, args.coefficient)
    common = {
        "seed": args.seed,
        "noise_frac": args.noise_frac,
        "device": args.device,
        "datasets_dir": str(args.datasets_dir),
        "results_base_folder": str(args.results_base_folder),
        "run_name": args.run_name,
    }
    if args.model == "mlp":
        from mnist_label_noise.config import MLPLabelNoiseConfig

        return MLPLabelNoiseConfig(
            **common,
            spectral_l1_reg_coef=regularization.spectral_coefficient,
            matrix_l2_reg_coef=regularization.l2_coefficient,
            hidden_dim=2048,
            n_hidden_layers=4,
            train_subset_size=3000,
            val_subset_size=5000,
            batch_size=128,
            epochs=60,
            lr=1e-3,
            beta1=0.9,
            beta2=0.95,
            skip_truncation_sweep=True,
            evaluate_test_each_epoch=False,
        )
    if args.model == "gru":
        from gru_label_noise.config import GRULabelNoiseConfig

        return GRULabelNoiseConfig(
            **common,
            spectral_l1_reg_coef=regularization.spectral_coefficient,
            matrix_l2_reg_coef=regularization.l2_coefficient,
            hidden_dim=1280,
            n_layers=2,
            train_subset_size=3000,
            val_subset_size=5000,
            batch_size=128,
            epochs=60,
            lr=1e-3,
            beta1=0.9,
            beta2=0.95,
            evaluate_test_each_epoch=False,
        )

    from bert_delta_label_noise.config import BertDeltaLabelNoiseConfig

    preset = BERT_DATASETS[args.dataset]
    return BertDeltaLabelNoiseConfig(
        **common,
        **preset,
        dataset=args.dataset,
        spectral_delta_coef=regularization.spectral_coefficient,
        matrix_l2_delta_coef=regularization.l2_coefficient,
        split_seed=args.split_seed,
        data_seed=args.data_seed,
        trainable_top_layers=4,
        regularize_classifier=False,
        batch_size=32,
        eval_batch_size=64,
        epochs=25,
        lr=2e-5,
        beta1=0.9,
        beta2=0.95,
        warmup_fraction=0.1,
        max_length=128,
    )


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=("mlp", "gru", "bert"))
    parser.add_argument("--method", required=True, choices=("no_wd", "l2", "spectral"))
    parser.add_argument("--coefficient", type=float, default=0.0)
    parser.add_argument("--noise-frac", type=float, required=True)
    parser.add_argument("--dataset", choices=tuple(BERT_DATASETS), default="ag_news")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=1337)
    parser.add_argument("--data-seed", type=int, default=2000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--datasets-dir", type=Path, default=Path.home() / "paper_datasets"
    )
    parser.add_argument(
        "--results-base-folder", type=Path, default=Path("exps/paper_robustness")
    )
    parser.add_argument("--run-name")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    cfg = build_config(args)
    if args.dry_run:
        print(json.dumps(dataclasses.asdict(cfg), indent=2))
        return
    if args.model == "mlp":
        from mnist_label_noise.train import train_one_run
    elif args.model == "gru":
        from gru_label_noise.train import train_one_run
    else:
        from bert_delta_label_noise.train import train_one_run
    train_one_run(cfg)


if __name__ == "__main__":
    main()
