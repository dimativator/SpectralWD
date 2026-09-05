import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen

import torch
from torch.utils.data import DataLoader, Dataset

from .config import BertDeltaLabelNoiseConfig


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def make_noisy_labels(
    clean_labels: torch.Tensor,
    noise_frac: float,
    num_labels: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not 0.0 <= noise_frac <= 1.0:
        raise ValueError("noise_frac must be in [0, 1]")
    if num_labels < 2:
        raise ValueError("num_labels must be at least 2")
    generator = torch.Generator().manual_seed(seed)
    count = round(len(clean_labels) * noise_frac)
    mask = torch.zeros(len(clean_labels), dtype=torch.bool)
    if count == 0:
        return clean_labels.clone(), mask
    indices = torch.randperm(len(clean_labels), generator=generator)[:count]
    mask[indices] = True
    offsets = torch.randint(
        1,
        num_labels,
        (count,),
        generator=generator,
    )
    noisy_labels = clean_labels.clone()
    noisy_labels[indices] = (clean_labels[indices] + offsets) % num_labels
    return noisy_labels, mask


class EncodedTextDataset(Dataset):
    def __init__(
        self,
        encodings: dict[str, torch.Tensor],
        clean_labels: torch.Tensor,
        noisy_labels: torch.Tensor,
        corruption_mask: torch.Tensor,
    ) -> None:
        self.encodings = encodings
        self.clean_labels = clean_labels
        self.noisy_labels = noisy_labels
        self.corruption_mask = corruption_mask

    def __len__(self) -> int:
        return len(self.clean_labels)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            **{name: values[index] for name, values in self.encodings.items()},
            "clean_label": self.clean_labels[index],
            "noisy_label": self.noisy_labels[index],
            "corruption_mask": self.corruption_mask[index],
        }


@dataclass(frozen=True)
class DataBundle:
    train_loader: DataLoader
    train_eval_loader: DataLoader
    validation_loader: DataLoader
    test_loader: DataLoader
    split_hash: str
    train_size: int
    validation_size: int
    test_size: int


def _balanced_split_indices(
    labels: list[int],
    train_size: int,
    validation_size: int,
    seed: int,
    num_labels: int = 4,
) -> tuple[list[int], list[int]]:
    if train_size % num_labels != 0 or validation_size % num_labels != 0:
        raise ValueError("Balanced split sizes must be divisible by num_labels")
    generator = torch.Generator().manual_seed(seed)
    train_per_class = train_size // num_labels
    validation_per_class = validation_size // num_labels
    train: list[int] = []
    validation: list[int] = []
    for label in range(num_labels):
        candidates = torch.tensor(
            [index for index, value in enumerate(labels) if value == label],
            dtype=torch.long,
        )
        order = torch.randperm(len(candidates), generator=generator)
        required = train_per_class + validation_per_class
        if len(candidates) < required:
            raise ValueError(f"Not enough unique examples for label {label}")
        shuffled = candidates[order[:required]].tolist()
        train.extend(shuffled[:train_per_class])
        validation.extend(shuffled[train_per_class:])
    train = torch.tensor(train)[
        torch.randperm(len(train), generator=generator)
    ].tolist()
    validation = torch.tensor(validation)[
        torch.randperm(len(validation), generator=generator)
    ].tolist()
    return train, validation


def _balanced_sample_indices(
    labels: list[int],
    size: int,
    seed: int,
    num_labels: int,
) -> list[int]:
    selected, empty = _balanced_split_indices(
        labels,
        size,
        0,
        seed,
        num_labels=num_labels,
    )
    if empty:
        raise AssertionError("Expected an empty validation sample")
    return selected


def make_confirmatory_ag_news_indices(
    labels: list[int],
    train_size: int,
    validation_size: int,
    test_size: int,
    split_seed: int,
    exclude_train_size: int,
    exclude_validation_size: int,
    exclude_split_seed: int,
) -> tuple[list[int], list[int], list[int], list[int]]:
    if test_size % 4 != 0:
        raise ValueError("Balanced AG News test size must be divisible by four")
    excluded_train, excluded_validation = _balanced_split_indices(
        labels,
        exclude_train_size,
        exclude_validation_size,
        exclude_split_seed,
    )
    excluded = set(excluded_train) | set(excluded_validation)
    generator = torch.Generator().manual_seed(split_seed)
    counts = (train_size // 4, validation_size // 4, test_size // 4)
    if train_size % 4 != 0 or validation_size % 4 != 0:
        raise ValueError("Balanced AG News split sizes must be divisible by four")
    train: list[int] = []
    validation: list[int] = []
    test: list[int] = []
    for label in range(4):
        candidates = torch.tensor(
            [
                index
                for index, value in enumerate(labels)
                if value == label and index not in excluded
            ],
            dtype=torch.long,
        )
        order = torch.randperm(len(candidates), generator=generator)
        required = sum(counts)
        if len(candidates) < required:
            raise ValueError(
                f"Not enough unused unique examples for AG News label {label}"
            )
        shuffled = candidates[order[:required]].tolist()
        first, second = counts[0], counts[0] + counts[1]
        train.extend(shuffled[:first])
        validation.extend(shuffled[first:second])
        test.extend(shuffled[second:])

    def shuffled(values: list[int]) -> list[int]:
        tensor = torch.tensor(values)
        return tensor[torch.randperm(len(tensor), generator=generator)].tolist()

    return shuffled(train), shuffled(validation), shuffled(test), sorted(excluded)


def _encode(tokenizer, texts: list[str], max_length: int) -> dict[str, torch.Tensor]:
    encoded = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return {name: value for name, value in encoded.items()}


def _build_bundle(
    cfg: BertDeltaLabelNoiseConfig,
    tokenizer,
    train_texts: list[str],
    train_clean: torch.Tensor,
    validation_texts: list[str],
    validation_clean: torch.Tensor,
    test_texts: list[str],
    test_clean: torch.Tensor,
    split_identity: bytes,
) -> DataBundle:
    train_noisy, corruption_mask = make_noisy_labels(
        train_clean,
        cfg.noise_frac,
        cfg.num_labels,
        cfg.data_seed,
    )
    zeros_validation = torch.zeros(len(validation_clean), dtype=torch.bool)
    zeros_test = torch.zeros(len(test_clean), dtype=torch.bool)
    train_dataset = EncodedTextDataset(
        _encode(tokenizer, train_texts, cfg.max_length),
        train_clean,
        train_noisy,
        corruption_mask,
    )
    validation_dataset = EncodedTextDataset(
        _encode(tokenizer, validation_texts, cfg.max_length),
        validation_clean,
        validation_clean,
        zeros_validation,
    )
    test_dataset = EncodedTextDataset(
        _encode(tokenizer, test_texts, cfg.max_length),
        test_clean,
        test_clean,
        zeros_test,
    )

    train_generator = torch.Generator().manual_seed(cfg.seed)
    common = {"num_workers": cfg.num_workers, "pin_memory": True}
    digest = hashlib.sha256()
    digest.update(split_identity)
    digest.update(train_noisy.numpy().tobytes())
    return DataBundle(
        train_loader=DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            generator=train_generator,
            **common,
        ),
        train_eval_loader=DataLoader(
            train_dataset,
            batch_size=cfg.eval_batch_size,
            shuffle=False,
            **common,
        ),
        validation_loader=DataLoader(
            validation_dataset,
            batch_size=cfg.eval_batch_size,
            shuffle=False,
            **common,
        ),
        test_loader=DataLoader(
            test_dataset,
            batch_size=cfg.eval_batch_size,
            shuffle=False,
            **common,
        ),
        split_hash=digest.hexdigest(),
        train_size=len(train_dataset),
        validation_size=len(validation_dataset),
        test_size=len(test_dataset),
    )


def _get_ag_news_data(cfg: BertDeltaLabelNoiseConfig, tokenizer) -> DataBundle:
    from datasets import load_dataset

    raw = load_dataset("fancyzhx/ag_news", cache_dir=cfg.datasets_dir)
    test_text_set = {_normalize_text(text) for text in raw["test"]["text"]}
    unique_texts: list[str] = []
    unique_labels: list[int] = []
    source_indices: list[int] = []
    seen: set[str] = set()
    for index, (text, label) in enumerate(
        zip(raw["train"]["text"], raw["train"]["label"])
    ):
        normalized = _normalize_text(text)
        if normalized in seen or normalized in test_text_set:
            continue
        seen.add(normalized)
        unique_texts.append(text)
        unique_labels.append(int(label))
        source_indices.append(index)

    train_indices, validation_indices = _balanced_split_indices(
        unique_labels,
        cfg.train_size,
        cfg.validation_size,
        cfg.split_seed,
    )

    train_texts = [unique_texts[index] for index in train_indices]
    validation_texts = [unique_texts[index] for index in validation_indices]
    train_clean = torch.tensor([unique_labels[index] for index in train_indices])
    validation_clean = torch.tensor(
        [unique_labels[index] for index in validation_indices]
    )
    test_clean = torch.tensor([int(label) for label in raw["test"]["label"]])
    split_identity = json.dumps(
        {
            "train": [source_indices[index] for index in train_indices],
            "validation": [source_indices[index] for index in validation_indices],
            "test": list(range(len(test_clean))),
        },
        sort_keys=True,
    ).encode()
    return _build_bundle(
        cfg,
        tokenizer,
        train_texts,
        train_clean,
        validation_texts,
        validation_clean,
        list(raw["test"]["text"]),
        test_clean,
        split_identity,
    )


def _get_ag_news_confirmatory_data(
    cfg: BertDeltaLabelNoiseConfig, tokenizer
) -> DataBundle:
    from datasets import load_dataset

    if cfg.exclude_split_seed < 0:
        raise ValueError("Confirmatory AG News requires exclude_split_seed")
    raw = load_dataset("fancyzhx/ag_news", cache_dir=cfg.datasets_dir)
    official_test_texts = {_normalize_text(text) for text in raw["test"]["text"]}
    unique_texts: list[str] = []
    unique_labels: list[int] = []
    source_indices: list[int] = []
    seen: set[str] = set()
    for index, (text, label) in enumerate(
        zip(raw["train"]["text"], raw["train"]["label"])
    ):
        normalized = _normalize_text(text)
        if normalized in seen or normalized in official_test_texts:
            continue
        seen.add(normalized)
        unique_texts.append(text)
        unique_labels.append(int(label))
        source_indices.append(index)

    train_indices, validation_indices, test_indices, excluded_indices = (
        make_confirmatory_ag_news_indices(
            unique_labels,
            cfg.train_size,
            cfg.validation_size,
            cfg.test_size,
            cfg.split_seed,
            cfg.exclude_train_size,
            cfg.exclude_validation_size,
            cfg.exclude_split_seed,
        )
    )
    train_texts = [unique_texts[index] for index in train_indices]
    validation_texts = [unique_texts[index] for index in validation_indices]
    test_texts = [unique_texts[index] for index in test_indices]
    train_clean = torch.tensor([unique_labels[index] for index in train_indices])
    validation_clean = torch.tensor(
        [unique_labels[index] for index in validation_indices]
    )
    test_clean = torch.tensor([unique_labels[index] for index in test_indices])
    split_identity = json.dumps(
        {
            "excluded_pilot": [source_indices[index] for index in excluded_indices],
            "train": [source_indices[index] for index in train_indices],
            "validation": [source_indices[index] for index in validation_indices],
            "test": [source_indices[index] for index in test_indices],
        },
        sort_keys=True,
    ).encode()
    return _build_bundle(
        cfg,
        tokenizer,
        train_texts,
        train_clean,
        validation_texts,
        validation_clean,
        test_texts,
        test_clean,
        split_identity,
    )


def _get_dbpedia14_data(cfg: BertDeltaLabelNoiseConfig, tokenizer) -> DataBundle:
    from datasets import load_dataset

    if cfg.num_labels != 14:
        raise ValueError("DBpedia-14 requires num_labels=14")
    raw = load_dataset("fancyzhx/dbpedia_14", cache_dir=cfg.datasets_dir)

    def combine_text(title_value, content_value) -> str:
        title = _normalize_text(str(title_value))
        content = _normalize_text(str(content_value))
        return f"{title}. {content}" if title else content

    test_text_column = [
        combine_text(title, content)
        for title, content in zip(raw["test"]["title"], raw["test"]["content"])
    ]
    official_test_texts = {_normalize_text(text) for text in test_text_column}
    train_texts_all: list[str] = []
    train_labels_all: list[int] = []
    train_sources: list[int] = []
    seen_train: set[str] = set()
    for index, (title, content, label) in enumerate(
        zip(raw["train"]["title"], raw["train"]["content"], raw["train"]["label"])
    ):
        text = combine_text(title, content)
        normalized = _normalize_text(text)
        if normalized in seen_train or normalized in official_test_texts:
            continue
        seen_train.add(normalized)
        train_texts_all.append(text)
        train_labels_all.append(int(label))
        train_sources.append(index)

    train_indices, validation_indices = _balanced_split_indices(
        train_labels_all,
        cfg.train_size,
        cfg.validation_size,
        cfg.split_seed,
        num_labels=cfg.num_labels,
    )

    test_texts_all: list[str] = []
    test_labels_all: list[int] = []
    test_sources: list[int] = []
    seen_test: set[str] = set()
    for index, (text, label) in enumerate(zip(test_text_column, raw["test"]["label"])):
        normalized = _normalize_text(text)
        if normalized in seen_test:
            continue
        seen_test.add(normalized)
        test_texts_all.append(text)
        test_labels_all.append(int(label))
        test_sources.append(index)
    test_indices = _balanced_sample_indices(
        test_labels_all,
        cfg.test_size,
        cfg.split_seed + 1,
        cfg.num_labels,
    )

    split_identity = json.dumps(
        {
            "dataset": "fancyzhx/dbpedia_14",
            "train": [train_sources[index] for index in train_indices],
            "validation": [train_sources[index] for index in validation_indices],
            "test": [test_sources[index] for index in test_indices],
        },
        sort_keys=True,
    ).encode()
    return _build_bundle(
        cfg,
        tokenizer,
        [train_texts_all[index] for index in train_indices],
        torch.tensor([train_labels_all[index] for index in train_indices]),
        [train_texts_all[index] for index in validation_indices],
        torch.tensor([train_labels_all[index] for index in validation_indices]),
        [test_texts_all[index] for index in test_indices],
        torch.tensor([test_labels_all[index] for index in test_indices]),
        split_identity,
    )


def _get_yahoo_answers_topics_data(
    cfg: BertDeltaLabelNoiseConfig, tokenizer
) -> DataBundle:
    from datasets import load_dataset

    if cfg.num_labels != 10:
        raise ValueError("Yahoo Answers Topics requires num_labels=10")
    raw = load_dataset("yahoo_answers_topics", cache_dir=cfg.datasets_dir)

    def combine_text(title, content, answer) -> str:
        parts = [
            _normalize_text(str(value))
            for value in (title, content, answer)
            if _normalize_text(str(value))
        ]
        return " [SEP] ".join(parts)

    def add_normalized_hash(batch):
        return {
            "_normalized_hash": [
                hashlib.sha256(
                    combine_text(title, content, answer).encode()
                ).hexdigest()
                for title, content, answer in zip(
                    batch["question_title"],
                    batch["question_content"],
                    batch["best_answer"],
                )
            ]
        }

    hashed = {
        name: split.map(
            add_normalized_hash,
            batched=True,
            batch_size=2000,
            num_proc=16,
            load_from_cache_file=True,
        )
        for name, split in raw.items()
    }
    official_test_hashes = set(hashed["test"]["_normalized_hash"])
    train_labels_all: list[int] = []
    train_sources: list[int] = []
    seen_train: set[str] = set()
    for index, (normalized_hash, label) in enumerate(
        zip(hashed["train"]["_normalized_hash"], raw["train"]["topic"])
    ):
        if normalized_hash in seen_train or normalized_hash in official_test_hashes:
            continue
        seen_train.add(normalized_hash)
        train_labels_all.append(int(label))
        train_sources.append(index)

    train_indices, validation_indices = _balanced_split_indices(
        train_labels_all,
        cfg.train_size,
        cfg.validation_size,
        cfg.split_seed,
        num_labels=cfg.num_labels,
    )

    test_labels_all: list[int] = []
    test_sources: list[int] = []
    seen_test: set[str] = set()
    for index, (normalized_hash, label) in enumerate(
        zip(hashed["test"]["_normalized_hash"], raw["test"]["topic"])
    ):
        if normalized_hash in seen_test:
            continue
        seen_test.add(normalized_hash)
        test_labels_all.append(int(label))
        test_sources.append(index)
    test_indices = _balanced_sample_indices(
        test_labels_all,
        cfg.test_size,
        cfg.split_seed + 1,
        cfg.num_labels,
    )

    split_identity = json.dumps(
        {
            "dataset": "yahoo_answers_topics",
            "train": [train_sources[index] for index in train_indices],
            "validation": [train_sources[index] for index in validation_indices],
            "test": [test_sources[index] for index in test_indices],
        },
        sort_keys=True,
    ).encode()

    def source_text(split: str, source_index: int) -> str:
        row = raw[split][source_index]
        return combine_text(
            row["question_title"], row["question_content"], row["best_answer"]
        )

    return _build_bundle(
        cfg,
        tokenizer,
        [source_text("train", train_sources[index]) for index in train_indices],
        torch.tensor([train_labels_all[index] for index in train_indices]),
        [source_text("train", train_sources[index]) for index in validation_indices],
        torch.tensor([train_labels_all[index] for index in validation_indices]),
        [source_text("test", test_sources[index]) for index in test_indices],
        torch.tensor([test_labels_all[index] for index in test_indices]),
        split_identity,
    )


def _get_yelp_review_full_data(cfg: BertDeltaLabelNoiseConfig, tokenizer) -> DataBundle:
    from datasets import load_dataset

    if cfg.num_labels != 5:
        raise ValueError("Yelp Review Full requires num_labels=5")
    raw = load_dataset("yelp_review_full", cache_dir=cfg.datasets_dir)

    def add_normalized_hash(batch):
        return {
            "_normalized_hash": [
                hashlib.sha256(_normalize_text(str(text)).encode()).hexdigest()
                for text in batch["text"]
            ]
        }

    hashed = {
        name: split.map(
            add_normalized_hash,
            batched=True,
            batch_size=2000,
            num_proc=16,
            load_from_cache_file=True,
        )
        for name, split in raw.items()
    }
    official_test_hashes = set(hashed["test"]["_normalized_hash"])
    train_labels_all: list[int] = []
    train_sources: list[int] = []
    seen_train: set[str] = set()
    for index, (normalized_hash, label) in enumerate(
        zip(hashed["train"]["_normalized_hash"], raw["train"]["label"])
    ):
        if normalized_hash in seen_train or normalized_hash in official_test_hashes:
            continue
        seen_train.add(normalized_hash)
        train_labels_all.append(int(label))
        train_sources.append(index)

    train_indices, validation_indices = _balanced_split_indices(
        train_labels_all,
        cfg.train_size,
        cfg.validation_size,
        cfg.split_seed,
        num_labels=cfg.num_labels,
    )

    test_labels_all: list[int] = []
    test_sources: list[int] = []
    seen_test: set[str] = set()
    for index, (normalized_hash, label) in enumerate(
        zip(hashed["test"]["_normalized_hash"], raw["test"]["label"])
    ):
        if normalized_hash in seen_test:
            continue
        seen_test.add(normalized_hash)
        test_labels_all.append(int(label))
        test_sources.append(index)
    test_indices = _balanced_sample_indices(
        test_labels_all,
        cfg.test_size,
        cfg.split_seed + 1,
        cfg.num_labels,
    )

    split_identity = json.dumps(
        {
            "dataset": "yelp_review_full",
            "train": [train_sources[index] for index in train_indices],
            "validation": [train_sources[index] for index in validation_indices],
            "test": [test_sources[index] for index in test_indices],
        },
        sort_keys=True,
    ).encode()
    return _build_bundle(
        cfg,
        tokenizer,
        [str(raw["train"][train_sources[index]]["text"]) for index in train_indices],
        torch.tensor([train_labels_all[index] for index in train_indices]),
        [
            str(raw["train"][train_sources[index]]["text"])
            for index in validation_indices
        ],
        torch.tensor([train_labels_all[index] for index in validation_indices]),
        [str(raw["test"][test_sources[index]]["text"]) for index in test_indices],
        torch.tensor([test_labels_all[index] for index in test_indices]),
        split_identity,
    )


def _get_banking77_data(cfg: BertDeltaLabelNoiseConfig, tokenizer) -> DataBundle:
    from datasets import load_dataset

    if cfg.num_labels != 77:
        raise ValueError("Banking77 requires num_labels=77")
    raw = load_dataset("banking77", cache_dir=cfg.datasets_dir)
    official_test_texts = {_normalize_text(str(text)) for text in raw["test"]["text"]}

    train_texts_all: list[str] = []
    train_labels_all: list[int] = []
    train_sources: list[int] = []
    seen_train: set[str] = set()
    for index, (text, label) in enumerate(
        zip(raw["train"]["text"], raw["train"]["label"])
    ):
        normalized = _normalize_text(str(text))
        if normalized in seen_train or normalized in official_test_texts:
            continue
        seen_train.add(normalized)
        train_texts_all.append(str(text))
        train_labels_all.append(int(label))
        train_sources.append(index)

    train_indices, validation_indices = _balanced_split_indices(
        train_labels_all,
        cfg.train_size,
        cfg.validation_size,
        cfg.split_seed,
        num_labels=cfg.num_labels,
    )

    test_texts_all: list[str] = []
    test_labels_all: list[int] = []
    test_sources: list[int] = []
    seen_test: set[str] = set()
    for index, (text, label) in enumerate(
        zip(raw["test"]["text"], raw["test"]["label"])
    ):
        normalized = _normalize_text(str(text))
        if normalized in seen_test:
            continue
        seen_test.add(normalized)
        test_texts_all.append(str(text))
        test_labels_all.append(int(label))
        test_sources.append(index)
    test_indices = _balanced_sample_indices(
        test_labels_all,
        cfg.test_size,
        cfg.split_seed + 1,
        cfg.num_labels,
    )

    split_identity = json.dumps(
        {
            "dataset": "banking77",
            "train": [train_sources[index] for index in train_indices],
            "validation": [train_sources[index] for index in validation_indices],
            "test": [test_sources[index] for index in test_indices],
        },
        sort_keys=True,
    ).encode()
    return _build_bundle(
        cfg,
        tokenizer,
        [train_texts_all[index] for index in train_indices],
        torch.tensor([train_labels_all[index] for index in train_indices]),
        [train_texts_all[index] for index in validation_indices],
        torch.tensor([train_labels_all[index] for index in validation_indices]),
        [test_texts_all[index] for index in test_indices],
        torch.tensor([test_labels_all[index] for index in test_indices]),
        split_identity,
    )


def _load_clinc_small(cfg: BertDeltaLabelNoiseConfig) -> dict[str, list[list[str]]]:
    cache_path = Path(cfg.datasets_dir).expanduser() / "clinc150" / "data_small.json"
    if cache_path.exists():
        payload = cache_path.read_bytes()
    else:
        with urlopen(cfg.clinc_data_url, timeout=60) as response:
            payload = response.read()
        digest = hashlib.sha256(payload).hexdigest()
        if digest != cfg.clinc_data_sha256:
            raise ValueError(
                f"CLINC150 SHA256 mismatch: expected {cfg.clinc_data_sha256}, got {digest}"
            )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    if digest != cfg.clinc_data_sha256:
        raise ValueError(
            f"Cached CLINC150 SHA256 mismatch: expected {cfg.clinc_data_sha256}, got {digest}"
        )
    return json.loads(payload)


def _get_clinc150_small_data(cfg: BertDeltaLabelNoiseConfig, tokenizer) -> DataBundle:
    if cfg.num_labels != 150:
        raise ValueError("CLINC150 requires num_labels=150")
    raw = _load_clinc_small(cfg)
    required_splits = {"train": 7500, "val": 3000, "test": 4500}
    for split, expected_size in required_splits.items():
        if split not in raw or len(raw[split]) != expected_size:
            actual_size = len(raw.get(split, []))
            raise ValueError(
                f"CLINC150 {split} size mismatch: expected {expected_size}, got {actual_size}"
            )

    labels = sorted({label for _, label in raw["train"]})
    if len(labels) != cfg.num_labels:
        raise ValueError(f"Expected 150 CLINC150 intents, got {len(labels)}")
    label_to_id = {label: index for index, label in enumerate(labels)}

    def unpack(split: str) -> tuple[list[str], torch.Tensor]:
        texts = [str(text) for text, _ in raw[split]]
        try:
            clean = torch.tensor([label_to_id[label] for _, label in raw[split]])
        except KeyError as error:
            raise ValueError(
                f"Unknown CLINC150 label in {split}: {error.args[0]}"
            ) from error
        return texts, clean

    train_texts, train_clean = unpack("train")
    validation_texts, validation_clean = unpack("val")
    test_texts, test_clean = unpack("test")
    split_identity = json.dumps(
        {
            "dataset_sha256": cfg.clinc_data_sha256,
            "labels": labels,
            "train": raw["train"],
            "validation": raw["val"],
            "test": raw["test"],
        },
        sort_keys=True,
    ).encode()
    return _build_bundle(
        cfg,
        tokenizer,
        train_texts,
        train_clean,
        validation_texts,
        validation_clean,
        test_texts,
        test_clean,
        split_identity,
    )


def get_data(cfg: BertDeltaLabelNoiseConfig, tokenizer) -> DataBundle:
    if cfg.dataset == "ag_news":
        return _get_ag_news_data(cfg, tokenizer)
    if cfg.dataset == "ag_news_confirmatory":
        return _get_ag_news_confirmatory_data(cfg, tokenizer)
    if cfg.dataset == "clinc150_small":
        return _get_clinc150_small_data(cfg, tokenizer)
    if cfg.dataset == "dbpedia_14":
        return _get_dbpedia14_data(cfg, tokenizer)
    if cfg.dataset == "yahoo_answers_topics":
        return _get_yahoo_answers_topics_data(cfg, tokenizer)
    if cfg.dataset == "yelp_review_full":
        return _get_yelp_review_full_data(cfg, tokenizer)
    if cfg.dataset == "banking77":
        return _get_banking77_data(cfg, tokenizer)
    raise ValueError(f"Unsupported dataset: {cfg.dataset}")
