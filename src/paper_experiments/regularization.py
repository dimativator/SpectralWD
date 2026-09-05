from dataclasses import dataclass


@dataclass(frozen=True)
class Regularization:
    method: str
    coefficient: float

    def __post_init__(self) -> None:
        if self.method not in {"no_wd", "l2", "spectral"}:
            raise ValueError(f"Unknown regularization method: {self.method}")
        if self.coefficient < 0:
            raise ValueError("coefficient must be non-negative")
        if self.method == "no_wd" and self.coefficient != 0:
            raise ValueError("no_wd requires coefficient=0")
        if self.method != "no_wd" and self.coefficient == 0:
            raise ValueError(f"{self.method} requires a positive coefficient")

    @property
    def spectral_coefficient(self) -> float:
        return self.coefficient if self.method == "spectral" else 0.0

    @property
    def l2_coefficient(self) -> float:
        return self.coefficient if self.method == "l2" else 0.0
