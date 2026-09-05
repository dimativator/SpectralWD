import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from .config import GRULabelNoiseConfig

NUM_CLASSES = 10


class NoisyMNISTSubset(Dataset):
    """Fixed uniform label replacements with the clean targets retained for analysis."""

    def __init__(self, base, indices, noise_frac: float, seed: int):
        self.base = base
        self.indices = list(indices)
        generator = torch.Generator().manual_seed(seed)
        n_noisy = int(len(self.indices) * noise_frac)
        permutation = torch.randperm(len(self.indices), generator=generator)
        self.is_noisy = torch.zeros(len(self.indices), dtype=torch.bool)
        self.is_noisy[permutation[:n_noisy]] = True
        self.random_labels = torch.randint(
            0, NUM_CLASSES, (len(self.indices),), generator=generator
        )
        self.original_labels = torch.as_tensor(
            [int(base.targets[index]) for index in self.indices], dtype=torch.long
        )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        image, clean_label = self.base[self.indices[index]]
        observed_label = (
            int(self.random_labels[index]) if self.is_noisy[index] else clean_label
        )
        return image, observed_label


def get_loaders(cfg: GRULabelNoiseConfig):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train_full = datasets.MNIST(
        root=cfg.datasets_dir, train=True, download=True, transform=transform
    )
    test_full = datasets.MNIST(
        root=cfg.datasets_dir, train=False, download=True, transform=transform
    )

    if cfg.train_subset_size + cfg.val_subset_size > len(train_full):
        raise ValueError("train_subset_size + val_subset_size exceeds MNIST train size")
    generator = torch.Generator().manual_seed(cfg.seed)
    indices = torch.randperm(len(train_full), generator=generator)
    train_indices = indices[: cfg.train_subset_size].tolist()
    val_indices = indices[
        cfg.train_subset_size: cfg.train_subset_size + cfg.val_subset_size
    ].tolist()
    train_subset = NoisyMNISTSubset(
        train_full, train_indices, cfg.noise_frac, seed=cfg.seed
    )

    train_loader = DataLoader(
        train_subset,
        batch_size=cfg.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(cfg.seed),
    )
    train_eval_loader = DataLoader(train_subset, batch_size=1024, shuffle=False)
    val_loader = DataLoader(
        Subset(train_full, val_indices), batch_size=1024, shuffle=False
    )
    test_loader = DataLoader(test_full, batch_size=1024, shuffle=False)
    return train_loader, train_eval_loader, val_loader, test_loader, train_subset
