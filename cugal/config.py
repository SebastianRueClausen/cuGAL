from enum import Enum
from dataclasses import dataclass
import torch

class SinkhornMethod(Enum):
    """The method used for Sinkhorn-Knopp."""

    STANDARD = 0
    """The standard Sinkhorn-Knopp method.
    This is faster, but usually requires double-precision floats.
    """

    LOG = 1
    """Perform Sinkhorn-Knopp in logarithmic space.
    Computational intensive but numerically stable and optimized for the GPU.
    Can be used with single-precision or half-precision floats.
    """

@dataclass
class Config:
    """Configuration of the CUGAL algorithm."""

    device: str = "cpu"
    """The torch device used for computations."""

    dtype: torch.dtype = torch.float64
    """The data type used for computations."""

    sinkhorn_regularization: float = 1.0
    sinkhorn_method: SinkhornMethod = SinkhornMethod.STANDARD
    sinkhorn_iterations: int = 500
    sinkhorn_threshold: float = 1e-9
    sinkhorn_eval_freq: int = 10

    mu: float = 0.5
    iter_count: int = 15

    def convert_tensor(self, input: torch.Tensor) -> torch.Tensor:
        return input.to(dtype=self.dtype, device=self.device)