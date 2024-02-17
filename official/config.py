from enum import Enum
from dataclasses import dataclass
import torch

class SinkhornMethod(Enum):
    STANDARD = 0
    LOG = 1

@dataclass
class Config:
    device: str
    sinkhorn_regularization: float
    sinkhorn_method: SinkhornMethod
    sinkhorn_iterations: int
    mu: float
    iter_count: int
    data_type: torch.dtype
