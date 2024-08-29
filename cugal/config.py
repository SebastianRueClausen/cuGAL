"""Configuration of cuGAL."""

from dataclasses import dataclass
from enum import Enum
from _collections_abc import Callable

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
    """

    MIX = 2
    """Start out in logarithmic space and switch over to standard
    Sinkhorn-Knopp when the cost matrix has stabilized.
    """


class HungarianMethod(Enum):
    """The method used for Hungarian algorithm."""

    SCIPY = 0
    """Use the scipy implementation of the Hungarian algorithm."""

    GREEDY = 1
    """Use the CUDA implementation of the Hungarian algorithm."""

    RAND = 2
    """Use the CUDA implementation of the Hungarian algorithm with random row order."""

    MORE_RAND = 3
    """Use the CUDA implementation of the Hungarian algorithm with random row order and distributed random column selection."""

    DOUBLE_GREEDY = 4
    """Use the CUDA implementation of the Hungarian algorithm with sorted row order and max column selection."""

    PARALLEL_GREEDY = 5
    """BEST_GREEDY with parallel computation of the assignements above 0.5"""


@dataclass
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

    lambda_func: Callable[[int], int] = lambda x: x
    """The function used to compute the regularization parameter."""

    def convert_tensor(self, input: torch.Tensor) -> torch.Tensor:
        """Convert tensor to correct type and dtype."""

        return input.to(dtype=self.dtype, device=self.device)
