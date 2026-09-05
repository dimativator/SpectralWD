import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class BertDeltaLabelNoiseConfig:
    seed: int = 0
    split_seed: int = 1337
    data_seed: int = 2000
    noise_frac: float = 0.6
    spectral_delta_coef: float = 0.0
    matrix_l2_delta_coef: float = 0.0

    dataset: str = "ag_news"
    model_name: str = "google-bert/bert-base-uncased"
    train_size: int = 5000
    validation_size: int = 5000
    test_size: int = 7600
    exclude_split_seed: int = -1
    exclude_train_size: int = 5000
    exclude_validation_size: int = 5000
    max_length: int = 128
    trainable_top_layers: int = 4
    num_labels: int = 4
    regularize_classifier: bool = False
    clinc_data_url: str = (
        "https://raw.githubusercontent.com/clinc/oos-eval/"
        "828f8093932c8fe6ca7936c3d2e52903b1c523de/data/data_small.json"
    )
    clinc_data_sha256: str = (
        "050e17476e6b4fa88f8518edaf09921c8f5e3a86dc8b63615361102a20b2ac01"
    )

    batch_size: int = 32
    eval_batch_size: int = 64
    epochs: int = 25
    lr: float = 2e-5
    beta1: float = 0.9
    beta2: float = 0.95
    warmup_fraction: float = 0.1
    grad_clip: float = 1.0
    num_workers: int = 2
    partition_eval_interval: int = 1

    device: str = "cuda"
    bf16: bool = True
    run_name: Optional[str] = None
    datasets_dir: str = field(
        default_factory=lambda: os.path.expanduser("~/ag_news_datasets")
    )
    results_base_folder: str = "./exps/bert_delta_label_noise_gate_20260830"
    skip_rank_analysis: bool = False
    evaluate_test: bool = True
    save_final_checkpoint: bool = False
