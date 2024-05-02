import torch
import torch.nn.functional as F
from cugal.config import Config, SinkhornMethod
from cugal.profile import SinkhornProfile, TimeStamp
from dataclasses import dataclass, field
import math

try:
    import cuda_kernels
    has_cuda = True
except ImportError:
    has_cuda = False

M_EPS = 1e-16


def initial_u(n: int, config: Config) -> torch.Tensor:
    match config.sinkhorn_method:
        case SinkhornMethod.LOG:
            return torch.zeros(n, device=config.device, dtype=config.dtype)
        case SinkhornMethod.MIX | SinkhornMethod.STANDARD:
            return torch.full(size=(n,), fill_value=1/n,
                              device=config.device, dtype=config.dtype)


@dataclass
class FixedInit:
    """Always init with same fixed u."""

    def get_u(self, K: torch.Tensor, config: Config) -> torch.Tensor:
        return initial_u(K.shape[0], config)

    def update(self, K: torch.Tensor, u: torch.Tensor):
        pass


@dataclass
class PrevInit:
    """Init with previous result."""

    previous: torch.Tensor | None = None

    def get_u(self, K: torch.Tensor, config: Config) -> torch.Tensor:
        return initial_u(K.shape[0], config) if self.previous is None else self.previous

    def update(self, K: torch.Tensor, u: torch.Tensor):
        self.previous = u


@dataclass
class CachedMatrix:
    row_sum: torch.Tensor
    col_sum: torch.Tensor

    u: torch.Tensor

    def simularity(self, row_sum: torch.Tensor, col_sum: torch.Tensor) -> torch.Tensor:
        return torch.mean((row_sum - self.row_sum)**2) + torch.mean(((col_sum - self.col_sum)**2))


@dataclass
class SelectiveInit:
    """Init by selecting among the previous n result."""

    cache_size: int = 5
    cached: list[CachedMatrix] = field(default_factory=list)

    def get_u(self, K: torch.Tensor, config: Config) -> torch.Tensor:
        if len(self.cached) == 0:
            return initial_u(K.shape[0], config)

        row_sum, col_sum = K.mean(1), K.mean(0)

        # TODO: Make sure this happens in parallel.
        simularities = [cached.simularity(
            row_sum, col_sum) for cached in self.cached]

        best_index = min(range(len(simularities)),
                         key=simularities.__getitem__)
        return self.cached[best_index].u

    def update(self, K: torch.Tensor, u: torch.Tensor):
        if len(self.cached) >= self.cache_size:
            self.cached.pop(0)

        self.cached.append(CachedMatrix(K.mean(1), K.mean(0), u))
        self.previous = u


SinkhornInit = FixedInit | PrevInit | SelectiveInit
"""Determines how Sinkhorn picks the initial guess for the scaling vectors."""


def init_from_cache_size(cache_size: int) -> SinkhornInit:
    match cache_size:
        case 0: return FixedInit()
        case 1: return PrevInit()
        case _: return SelectiveInit(cache_size=cache_size)


def can_use_cuda(config: Config) -> bool:
    return has_cuda and "cuda" in config.device and config.dtype in [
        torch.float32, torch.float16]


def is_close(a: torch.Tensor, b: torch.Tensor, config: Config) -> float:
    return torch.allclose(a, b, rtol=0.05, atol=config.sinkhorn_threshold) 


def is_close_log(log_a: torch.Tensor, log_b: torch.Tensor, config: Config) -> float:
    log_a, log_b = log_a.to(dtype=torch.float64), log_b.to(dtype=torch.float64)
    return is_close(log_a.exp(), log_b.exp(), config)


def relative_difference(a: torch.Tensor, b: torch.Tensor) -> float:
    return (abs(a - b).max() / max(abs(a).max(), abs(b).max(), 1)).item()


def sinkhorn(
    C: torch.Tensor,
    config: Config,
    profile=SinkhornProfile(),
    start: SinkhornInit = FixedInit(),
) -> torch.Tensor:
    match config.sinkhorn_method:
        case SinkhornMethod.STANDARD:
            return sinkhorn_knopp(C, config, profile, start)
        case SinkhornMethod.LOG:
            return loghorn(C, config, profile, start)
        case SinkhornMethod.MIX:
            return mixhorn(C, config, profile, start)
        case SinkhornMethod.OT_CPU:
            return sinkhorn_OT_cpu(C, config, profile, start)
            
def sinkhorn_OT_cpu(
    C: torch.Tensor,
    config: Config,
    profile=SinkhornProfile(),
    start: SinkhornInit = FixedInit(),
) -> torch.Tensor:
    import ot
    start_time = TimeStamp(config.device)
    ones = torch.ones(C.shape[0], device=config.device, dtype=config.dtype)
    output = ot.sinkhorn(ones, ones, C.cpu(), config.sinkhorn_regularization)
    profile.time = TimeStamp(config.device).elapsed_seconds(start_time)
    return output


def sinkhorn_knopp(
    C: torch.Tensor,
    config: Config,
    profile=SinkhornProfile(),
    start: SinkhornInit = FixedInit(),
) -> torch.Tensor:
    start_time = TimeStamp(config.device)

    K = torch.exp(C / -config.sinkhorn_regularization)
    u = start.get_u(K, config)

    for iteration in range(config.sinkhorn_iterations):
        prev_v = v if iteration != 0 else u
        prev_u = u

        v = 1 / (u @ K + M_EPS)
        u = 1 / (K @ v + M_EPS)

        if iteration % config.sinkhorn_eval_freq == 0:
            if is_close(u, prev_u, config) and is_close(v, prev_v, config):
                break

    output = u.reshape(-1, 1) * K * v.reshape(1, -1)
    start.update(K, u)

    profile.iteration_count = iteration + 1
    profile.time = TimeStamp(config.device).elapsed_seconds(start_time)

    return output


def loghorn(
    C: torch.Tensor,
    config: Config,
    profile=SinkhornProfile(),
    start: SinkhornInit = FixedInit,
) -> torch.Tensor:
    start_time = TimeStamp(config.device)

    K = - C / config.sinkhorn_regularization

    use_cuda = can_use_cuda(config)
    if use_cuda:
        K_transpose = K.t().contiguous()

    u = start.get_u(K, config)
    v = torch.zeros(K.shape[1], device=config.device, dtype=config.dtype)

    for iteration in range(config.sinkhorn_iterations):
        prev_v = v
        prev_u = u

        if use_cuda:
            cuda_kernels.sinkhorn_step(K_transpose, u, v)
            cuda_kernels.sinkhorn_step(K, v, u)
        else:
            v = -torch.logsumexp(K + u[:, None], 0)
            u = -torch.logsumexp(K + v[None, :], 1)

        if iteration % config.sinkhorn_eval_freq == 0:
            if is_close_log(u, prev_u, config) + is_close_log(v, prev_v, config):
            #if relative_difference(u, prev_u) + relative_difference(v, prev_v) < config.sinkhorn_threshold * 2:
                break

    output = torch.exp(K + u[:, None] + v[None, :])
    start.update(K, u)

    profile.iteration_count = iteration + 1
    profile.time = TimeStamp(config.device).elapsed_seconds(start_time)

    return output


def mixhorn(
    C: torch.Tensor,
    config: Config,
    profile=SinkhornProfile(),
    start: SinkhornInit = FixedInit,
) -> torch.Tensor:
    n, _ = C.shape

    start_time = TimeStamp(config.device)

    C /= -config.sinkhorn_regularization
    K = C

    v = torch.logsumexp(K, 0)
    K = torch.exp(K - v[None, :])

    u = start.get_u(K, config)

    for iteration in range(config.sinkhorn_iterations):
        prev_v = v if iteration != 0 else u
        prev_u = u

        v = 1 / (u @ K + M_EPS)
        u = 1 / (K @ v + M_EPS)

        if iteration % config.sinkhorn_eval_freq == 0:
            if is_close(u, prev_u, config) and is_close(v, prev_v, config):
                break

    K *= v.reshape(1, -1)
    K *= u.reshape(-1, 1)
    output = K

    start.update(K, u)

    profile.iteration_count = iteration + 1
    profile.time = TimeStamp(config.device).elapsed_seconds(start_time)

    return output
