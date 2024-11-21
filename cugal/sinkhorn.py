import torch
from cugal.config import Config, SinkhornMethod
from cugal.profile import SinkhornProfile, TimeStamp
from dataclasses import dataclass, field
from typing import Optional, TypeAlias
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


SinkhornInit: TypeAlias = FixedInit | PrevInit | SelectiveInit


def init_from_cache_size(cache_size: int) -> SinkhornInit:
    match cache_size:
        case 0: return FixedInit()
        case 1: return PrevInit()
        case _: return SelectiveInit(cache_size=cache_size)


def can_use_cuda(config: Config) -> bool:
    return has_cuda and "cuda" in config.device and config.dtype == torch.float32


def relative_difference(a: torch.Tensor, b: torch.Tensor) -> float:
    return (abs(a - b).max() / max(abs(a).max(), abs(b).max(), 1)).item()


def relative_difference_log(a: torch.Tensor, b: torch.Tensor) -> float:
    return relative_difference(a.to(dtype=torch.float64).exp(), b.to(dtype=torch.float64).exp())


def sinkhorn(
    C: torch.Tensor,
    config: Config,
    profile=SinkhornProfile(),
    start: SinkhornInit = FixedInit(),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    match config.sinkhorn_method:
        case SinkhornMethod.STANDARD:
            return sinkhorn_knopp(C, config, profile, start)
        case SinkhornMethod.LOG:
            return loghorn(C, config, profile, start)
        case SinkhornMethod.MIX:
            return mixhorn(C, config, profile, start)


def scale_kernel_matrix(K: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    K *= v.reshape(1, -1)
    K *= u.reshape(-1, 1)
    return K


def scale_kernel_matrix_log(K: torch.Tensor, log_u: torch.Tensor, log_v: torch.Tensor) -> torch.Tensor:
    K += log_u[:, None]
    K += log_v[None, :]
    torch.exp(K, out=K)
    return K


def marginal_error(K: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> float:
    return torch.sum(abs(torch.sum(scale_kernel_matrix(K.clone(), u, v), dim=0) - 1)).item()


def marginal_error_log(K: torch.Tensor, log_u: torch.Tensor, log_v: torch.Tensor) -> float:
    return torch.sum(abs(torch.sum(scale_kernel_matrix_log(K.clone(), log_u, log_v), dim=0) - 1)).item()


def momentum_weight(errors: list[float]) -> Optional[float]:
    if len(errors) < 2:
        return None
    ratio = min(errors[-1] / errors[-2], 0.99)
    if math.isnan(ratio):
        ratio = 0.99
    return 2 / (1 + math.sqrt(1 - ratio))


def sinkhorn_knopp(
    C: torch.Tensor,
    config: Config,
    profile=SinkhornProfile(),
    start: SinkhornInit = FixedInit(),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    start_time = TimeStamp(config.device)

    K = torch.exp(C / -config.sinkhorn_regularization)
    u = start.get_u(K, config)

    errors = []
    w = 1.0

    for iteration in range(config.sinkhorn_iterations):
        prev_v = v if iteration != 0 else u
        prev_u = u

        v = prev_v ** (1 - w) * (1 / (u @ K + M_EPS)) ** w
        u = prev_u ** (1 - w) * (1 / (K @ v + M_EPS)) ** w

        if iteration % config.sinkhorn_eval_freq == 0:
            error = marginal_error(K, u, v)
            if relative_difference(u, prev_u) + relative_difference(v, prev_v) < config.sinkhorn_threshold * 2:
                break
            errors.append(error)

            w_new = momentum_weight(errors)
            if not w_new is None:
                if config.frank_wolfe_threshold is None:
                    w = w_new

    start.update(K, u)

    profile.iteration_count = iteration + 1
    profile.time = TimeStamp(config.device).elapsed_seconds(start_time)

    return K * config.sinkhorn_regularization, u, v


def loghorn(
    C: torch.Tensor,
    config: Config,
    profile=SinkhornProfile(),
    start: SinkhornInit = FixedInit(),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    start_time = TimeStamp(config.device)

    if config.dynamic_sinkhorn_regularization:
        sinkhorn_reg = C.abs().max() / config.sinkhorn_scaling
        K = - C / sinkhorn_reg
    else:
        K = - C / config.sinkhorn_regularization
    #print(K.abs().max())

    use_cuda = can_use_cuda(config)
    if use_cuda:
        K_transpose = K.t().contiguous()

    u = start.get_u(K, config)
    v = torch.zeros(K.shape[1], device=config.device, dtype=config.dtype)

    errors = []
    w = 1

    for iteration in range(config.sinkhorn_iterations):
        prev_v = torch.clone(v)
        prev_u = torch.clone(u)

        if use_cuda:
            cuda_kernels.sinkhorn_log_step(K_transpose, u, v)
            cuda_kernels.sinkhorn_log_step(K, v, u)
        else:
            v = v * (1 - w) - w * torch.logsumexp(K + u[:, None], 0)
            u = u * (1 - w) - w * torch.logsumexp(K + v[None, :], 1)

        if iteration % config.sinkhorn_eval_freq == 0 and iteration != 0:
            error = marginal_error_log(K, u, v)
            if relative_difference_log(u, prev_u) + relative_difference_log(v, prev_v) < config.sinkhorn_threshold * 2:
                break
            errors.append(error)

            w_new = momentum_weight(errors)
            if not w_new is None:
                if config.frank_wolfe_threshold is None:
                    print(w_new)
                    w = w_new

    start.update(K, u)

    profile.iteration_count = iteration + 1
    profile.time = TimeStamp(config.device).elapsed_seconds(start_time)

    return K, u, v


def mixhorn(
    C: torch.Tensor,
    config: Config,
    profile=SinkhornProfile(),
    start: SinkhornInit = FixedInit(),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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

        if relative_difference(u, prev_u) + relative_difference(v, prev_v) < config.sinkhorn_threshold * 2:
            break

    start.update(K, u)

    profile.iteration_count = iteration + 1
    profile.time = TimeStamp(config.device).elapsed_seconds(start_time)

    return K, u, v
