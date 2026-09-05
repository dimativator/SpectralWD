from .config import GRULabelNoiseConfig

__all__ = ["GRULabelNoiseConfig", "RowSequentialGRU"]


def __getattr__(name):
    if name == "RowSequentialGRU":
        from .model import RowSequentialGRU

        return RowSequentialGRU
    raise AttributeError(name)
