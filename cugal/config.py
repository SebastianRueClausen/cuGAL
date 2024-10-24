"""Configuration of cuGAL."""

from dataclasses import dataclass
import dataclasses
from enum import Enum

import torch


class SinkhornMethod(str, Enum):
    """The method used for Sinkhorn-Knopp."""

    STANDARD = "STANDARD"
    """The standard Sinkhorn-Knopp method.
    This is faster, but usually requires double-precision floats.
    """

    LOG = "LOG"
    """Perform Sinkhorn-Knopp in logarithmic space.
    Computational intensive but numerically stable and optimized for the GPU.
    """

    MIX = "MIX"
    """Start out in logarithmic space and switch over to standard
    Sinkhorn-Knopp when the cost matrix has stabilized.
    """


class HungarianMethod(Enum):
    """The method used for Hungarian algorithm."""

    SCIPY = "SCIPY"
    """Use the scipy implementation of the Hungarian algorithm."""

    GREEDY = "GREEDY"
    """Use the CUDA implementation of the Hungarian algorithm."""

    RAND = "RAND"
    """Use the CUDA implementation of the Hungarian algorithm with random row order."""

    MORE_RAND = "MORE_RAND"
    """Use the CUDA implementation of the Hungarian algorithm with random row order and distributed random column selection."""

    DOUBLE_GREEDY = "DOUBLE_GREEDY"
    """Use the CUDA implementation of the Hungarian algorithm with sorted row order and max column selection."""

    PARALLEL_GREEDY = "PARALLEL_GREEDY"
    """BEST_GREEDY with parallel computation of the assignements above 0.5"""

    DENSE = "DENSE"
    SPARSE = "SPARSE"


@dataclass(frozen=True)
class Config:
    """Configuration of the CUGAL algorithm."""

    safe_mode: bool = False
    """If true, the algorithm will check for NaNs and Infs in the cost matrix."""

    device: str = "cpu"
    """The torch device used for computations."""

    dtype: torch.dtype = torch.float64
    """The data type used for computations."""

    sinkhorn_regularization: float = 1.0
    """Regularization of the cost matrix when running Sinkhorn.

    Higher values can help with numeric stability, but can lower accuracy."""

    sinkhorn_method: SinkhornMethod = SinkhornMethod.STANDARD
    """The version of Sinkhorn used."""

    sinkhorn_iterations: int = 500
    """The maximum number of sinkhorn iterations performed."""

    sinkhorn_threshold: float = 1e-3
    """The error threshold tolerated when running Sinkhorn."""

    sinkhorn_eval_freq: int = 10
    """How many Sinhorn iterations performed between checking for the potential of stopping."""

    mu: float = 0.5
    """The contribution of node features in finding the alignment."""

    iter_count: int = 15
    """The number of iterations to perform."""

    frank_wolfe_iter_count: int = 10
    """The number of Frank-Wolfe iterations to perform."""

    frank_wolfe_threshold: float | None = None
    """The max difference of the objective before stopping when running Frank-Wolfe."""

    use_sparse_adjacency: bool = False
    """Use sparse matrix representation for adjacency matrices."""

    sinkhorn_cache_size: int = 0
    """The size of the cache used to warm-start Sinkhorn."""

    recompute_distance: bool = False
    """Avoid storing distance matrix by doing recalculating each iteration."""

    hungarian_method: HungarianMethod = HungarianMethod.DOUBLE_GREEDY
    """The version of Hungarian algorithm used."""

    def convert_tensor(self, input: torch.Tensor) -> torch.Tensor:
        """Convert tensor to correct type and dtype."""

        return input.to(dtype=self.dtype, device=self.device)

    def to_dict(self) -> dict:
        config_dict = dataclasses.asdict(self)
        config_dict['dtype'] = str(self.dtype).removeprefix("torch.")
        config_dict['sinkhorn_method'] = self.sinkhorn_method.value
        config_dict['hungarian_method'] = self.hungarian_method.value
        return config_dict

    @classmethod
    def from_dict(cls, config_dict: dict):
        config_dict['dtype'] = getattr(torch, config_dict['dtype'])
        config_dict['sinkhorn_method'] = SinkhornMethod(
            config_dict['sinkhorn_method'])
        config_dict['hungarian_method'] = HungarianMethod[config_dict['hungarian_method']]
        return cls(**config_dict)
