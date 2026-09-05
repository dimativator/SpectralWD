from .config import MLPLabelNoiseConfig

__all__ = [
    "MLPLabelNoiseConfig",
    "NoisyLabelSubset",
    "get_loaders",
    "MLP",
    "get_model",
    "train_one_run",
]


def __getattr__(name):
    if name in {"NoisyLabelSubset", "get_loaders"}:
        from .data import NoisyLabelSubset, get_loaders

        return {"NoisyLabelSubset": NoisyLabelSubset, "get_loaders": get_loaders}[name]
    if name in {"MLP", "get_model"}:
        from .model import MLP, get_model

        return {"MLP": MLP, "get_model": get_model}[name]
    if name == "train_one_run":
        from .train import train_one_run

        return train_one_run
    raise AttributeError(name)
