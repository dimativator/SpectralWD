import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class GRULabelNoiseConfig:
    seed: int = 0
    noise_seed: Optional[int] = None
    spectral_l1_reg_coef: float = 0.0
    matrix_l2_reg_coef: float = 0.0
    spectral_wd_order: str = "post"
    l2_wd_order: str = "pre"
    noise_frac: float = 0.6

    hidden_dim: int = 1280
    n_layers: int = 2
    train_subset_size: int = 3000
    val_subset_size: int = 5000
    batch_size: int = 128
    epochs: int = 60
    lr: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.95

    device: str = "cpu"
    run_name: Optional[str] = None
    datasets_dir: str = field(default_factory=lambda: os.path.expanduser("~/mnist_datasets"))
    results_base_folder: str = "./exps/gru_rowseq_mnist_label_noise_matched"
    skip_rank_analysis: bool = False
    evaluate_test_each_epoch: bool = False
