import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class MLPLabelNoiseConfig:
    seed: int = 0
    noise_seed: Optional[int] = None
    spectral_l1_reg_coef: float = 0.0
    matrix_l2_reg_coef: float = 0.0
    spectral_wd_order: str = "post"
    l2_wd_order: str = "pre"
    noise_frac: float = 0.25

    hidden_dim: int = 2048
    n_hidden_layers: int = 4

    train_subset_size: int = 3000
    val_subset_size: int = 0
    batch_size: int = 128
    epochs: int = 60
    lr: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.95

    device: str = "cpu"

    wandb: bool = False
    wandb_project: str = "spectral-wd"
    wandb_entity: Optional[str] = None
    run_name: Optional[str] = None

    # Deliberately not "./datasets" or reading a DATASETS_DIR env var: that name/path
    # is used elsewhere in this repo for the shared fineweb-edu corpus, and MNIST must
    # never be downloaded there (often read-only/shared). Home directory is always writable.
    datasets_dir: str = field(default_factory=lambda: os.path.expanduser("~/mnist_datasets"))
    results_base_folder: str = "./exps/mnist_label_noise"
    skip_truncation_sweep: bool = False
    skip_rank_analysis: bool = False
    evaluate_test_each_epoch: bool = False
