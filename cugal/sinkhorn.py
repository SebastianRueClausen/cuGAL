import torch
from cugal.config import Config, SinkhornMethod
from cugal.profile import SinkhornProfile, TimeStamp
from dataclasses import dataclass
from enum import Enum
import math

try:
    import cuda_kernels
    has_cuda = True
except ImportError:
    has_cuda = False

M_EPS = 1e-16


class SinkhornResult(Enum):
    """Determines the output of running Sinkhorn."""

    PLAN = 0
    """Return the optimal transport plan."""

    SCALING = 1
    """Return the u, v scaling vectors."""


@dataclass
class SinkhornCache:
    """A cache of previous result from running Sinkhorn.

    This is used to do a warm start to subsequent Sinkhorn runs on similar matrices."""

    u_guess: torch.Tensor | None = None

    def update(self, u: torch.Tensor):
        self.u_guess = u


def can_use_cuda(config: Config) -> bool:
    return has_cuda and "cuda" in config.device and config.dtype in [
        torch.float32, torch.float16]


def relative_difference(a: torch.Tensor, b: torch.Tensor) -> float:
    return (abs(a - b).max() / max(abs(a).max(), abs(b).max(), 1)).item()


def sinkhorn(
    C: torch.Tensor,
    config: Config,
    profile=SinkhornProfile(),
    cache: SinkhornCache | None = None,
    result: SinkhornResult = SinkhornResult.PLAN,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    match config.sinkhorn_method:
        case SinkhornMethod.STANDARD:
            return sinkhorn_knopp(C, config, profile, cache, result)
        case SinkhornMethod.LOG:
            return loghorn(C, config, profile, cache, result)
        case SinkhornMethod.MIX:
            return mixhorn(C, config, profile, cache, result)


def sinkhorn_knopp(
    C: torch.Tensor,
    config: Config,
    profile=SinkhornProfile(),
    cache: SinkhornCache | None = None,
    result: SinkhornResult = SinkhornResult.PLAN,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    start_time = TimeStamp(config.device)

    na, _ = C.shape

    if cache is None or cache.u_guess is None:
        u = torch.full(size=(na,), fill_value=1/na,
                       device=config.device, dtype=config.dtype)
    else:
        u = torch.clone(cache.u_guess)

    K = torch.exp(C / -config.sinkhorn_regularization)

    for iteration in range(config.sinkhorn_iterations):
        prev_v = v if iteration != 0 else u
        prev_u = u

        v = 1 / (u @ K + M_EPS)
        u = 1 / (K @ v + M_EPS)

        if iteration % config.sinkhorn_eval_freq == 0:
            if (
                relative_difference(u, prev_u) + relative_difference(v, prev_v)
                    < config.sinkhorn_threshold * 2
            ):
                break

    match result:
        case SinkhornResult.PLAN:
            output = u.reshape(-1, 1) * K * v.reshape(1, -1)
        case SinkhornResult.SCALING:
            output = u, v

    if not cache is None:
        cache.update(u)

    profile.iteration_count = iteration + 1
    profile.time = TimeStamp(config.device).elapsed_seconds(start_time)

    return output


def loghorn(
    C: torch.Tensor,
    config: Config,
    profile=SinkhornProfile(),
    cache: SinkhornCache | None = None,
    result: SinkhornResult = SinkhornResult.PLAN,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    start_time = TimeStamp(config.device)

    use_cuda = can_use_cuda(config)
    na, nb = C.shape
    K = - C / config.sinkhorn_regularization

    if use_cuda:
        K_transpose = K.t().contiguous()

    if cache is None or cache.u_guess is None:
        u = torch.zeros(na, device=config.device, dtype=config.dtype)
    else:
        u = torch.clone(cache.u_guess)

    v = torch.zeros(nb, device=config.device, dtype=config.dtype)

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
            # TODO: Figure out how to handle this correctly.
            if (
                relative_difference(u, prev_u) + relative_difference(v, prev_v)
                    < math.log(config.sinkhorn_threshold * 2)
            ):
                break

    match result:
        case SinkhornResult.PLAN:
            output = torch.exp(K + u[:, None] + v[None, :])
        case SinkhornResult.SCALING:
            output = u, v

    if not cache is None:
        cache.update(u)

    profile.iteration_count = iteration + 1
    profile.time = TimeStamp(config.device).elapsed_seconds(start_time)

    return output


def mixhorn(
    C: torch.Tensor,
    config: Config,
    profile=SinkhornProfile(),
    cache: SinkhornCache | None = None,
    result: SinkhornResult = SinkhornResult.PLAN,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    n, _ = C.shape

    start_time = TimeStamp(config.device)

    K = -C / config.sinkhorn_regularization

    v = torch.logsumexp(K, 0)
    K = torch.exp(K - v[None, :])

    if cache is None or cache.u_guess is None:
        u = torch.full((n,), 1/n, device=config.device, dtype=config.dtype)
    else:
        u = cache.u_guess

    for iteration in range(config.sinkhorn_iterations):
        prev_v = v if iteration != 0 else u
        prev_u = u

        v = 1 / (u @ K + M_EPS)
        u = 1 / (K @ v + M_EPS)

        if iteration % config.sinkhorn_eval_freq == 0:
            if (
                relative_difference(u, prev_u) + relative_difference(v, prev_v)
                    < config.sinkhorn_threshold * 2
            ):
                break

    match result:
        case SinkhornResult.PLAN:
            output = u.reshape(-1, 1) * K * v.reshape(1, -1)
        case SinkhornResult.SCALING:
            output = u, v

    if not cache is None:
        cache.update(u)

    profile.iteration_count = iteration + 1
    profile.time = TimeStamp(config.device).elapsed_seconds(start_time)

    return output
