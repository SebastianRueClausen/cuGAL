from enum import Enum
from dataclasses import dataclass
import torch

class SinkhornMethod(Enum):
    STANDARD = 0
    LOG = 1

@dataclass
class Config:
    device: str = "cpu"
    sinkhorn_regularization: float = 1.0
    sinkhorn_method: SinkhornMethod = SinkhornMethod.STANDARD
    sinkhorn_iterations: int = 500
    sinkhorn_threshold: float = 1e-9
    sinkhorn_eval_freq: int = 10
    mu: float = 0.5
    iter_count: int = 15
    data_type: torch.dtype = torch.float64
