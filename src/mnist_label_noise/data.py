import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from .config import MLPLabelNoiseConfig

NUM_CLASSES = 10


class NoisyLabelSubset(Dataset):
    """Wraps a base dataset, replacing `noise_frac` of the labels (at the given
    indices) with an i.i.d. uniform random label, fixed once at construction so
    a given training point's noisiness never changes across epochs -- the
    direct analogue of a corrupted image label, not resampled noise."""

    def __init__(self, base, indices, noise_frac: float, num_classes: int, seed: int):
        self.base = base
        self.indices = indices
        g = torch.Generator().manual_seed(seed)
        n_noisy = int(len(indices) * noise_frac)
        perm = torch.randperm(len(indices), generator=g)
        self.is_noisy = torch.zeros(len(indices), dtype=torch.bool)
        self.is_noisy[perm[:n_noisy]] = True
        self.random_labels = torch.randint(0, num_classes, (len(indices),), generator=g)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        x, y = self.base[self.indices[i]]
        if self.is_noisy[i]:
            y = int(self.random_labels[i])
        return x, y


def get_loaders(cfg: MLPLabelNoiseConfig):
    tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_full = datasets.MNIST(root=cfg.datasets_dir, train=True, download=True, transform=tfm)
    test_full = datasets.MNIST(root=cfg.datasets_dir, train=False, download=True, transform=tfm)

    g = torch.Generator().manual_seed(cfg.seed)
    if cfg.train_subset_size + cfg.val_subset_size > len(train_full):
        raise ValueError("train_subset_size + val_subset_size exceeds MNIST train size")
    perm = torch.randperm(len(train_full), generator=g)
    idx = perm[: cfg.train_subset_size].tolist()
    val_idx = perm[
        cfg.train_subset_size: cfg.train_subset_size + cfg.val_subset_size
    ].tolist()
    train_subset = NoisyLabelSubset(train_full, idx, cfg.noise_frac, NUM_CLASSES, seed=cfg.seed)

    train_loader = DataLoader(train_subset, batch_size=cfg.batch_size, shuffle=True)
    train_eval_loader = DataLoader(train_subset, batch_size=1024, shuffle=False)
    val_loader = (
        DataLoader(Subset(train_full, val_idx), batch_size=1024, shuffle=False)
        if val_idx
        else None
    )
    test_loader = DataLoader(test_full, batch_size=1024, shuffle=False)
    return train_loader, train_eval_loader, val_loader, test_loader, train_subset
