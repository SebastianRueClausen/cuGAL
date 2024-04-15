import torch
from cugal.config import Config, SinkhornMethod
from cugal.profile import SinkhornProfile, TimeStamp
from dataclasses import dataclass

try:
    import cuda_kernels
    has_cuda = True
except ImportError:
    has_cuda = False

M_EPS = 1e-16


@dataclass
class SinkhornCache:
    u_guess: torch.Tensor | None = None

    def update(self, u: torch.Tensor):
        if self.u_guess is None:
            self.u_guess = u
        else:
            diff = u - self.u_guess
            self.u_guess = u - torch.clamp(diff, -1.0, 1.0)


def can_use_cuda(config: Config) -> bool:
    return has_cuda and "cuda" in config.device and config.dtype in [
        torch.float32, torch.float16]


def sinkhorn(
    C: torch.Tensor,
    config: Config,
    profile=SinkhornProfile(),
    cache: SinkhornCache | None = None,
) -> torch.Tensor:
    match config.sinkhorn_method:
        case SinkhornMethod.STANDARD:
            return sinkhorn_knopp(C, config, profile, cache)
        case SinkhornMethod.LOG:
            return loghorn(C, config, profile, cache)
        case SinkhornMethod.MIX:
            return mixhorn(C, config, profile, cache)


def sinkhorn_knopp(
    C: torch.Tensor,
    config: Config,
    profile=SinkhornProfile(),
    cache: SinkhornCache | None = None,
) -> torch.Tensor:
    start_time = TimeStamp(config.device)

    na, _ = C.shape

    if cache is None or cache.u_guess is None:
        u = torch.full(size=(na,), fill_value=1/na,
                       device=config.device, dtype=config.dtype)
    else:
        u = torch.clone(cache.u_guess)

    K = torch.exp(C / -config.sinkhorn_regularization)

    for iteration in range(config.sinkhorn_iterations):
        v = 1 / (u @ K + M_EPS)
        u = 1 / (K @ v + M_EPS)

        if iteration % config.sinkhorn_eval_freq == 0:
            b_hat = torch.matmul(u, K) * v
            threshold = (1 - b_hat).pow(2).sum().item()
            profile.errors.append(threshold)
            if threshold < config.sinkhorn_threshold:
                break

    output = u.reshape(-1, 1) * K * v.reshape(1, -1)

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
) -> torch.Tensor:
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
        if use_cuda:
            cuda_kernels.sinkhorn_step(K_transpose, u, v)
            cuda_kernels.sinkhorn_step(K, v, u)
        else:
            v = -torch.logsumexp(K + u[:, None], 0)
            u = -torch.logsumexp(K + v[None, :], 1)

        if iteration % config.sinkhorn_eval_freq == 0:
            tmp = torch.sum(torch.exp(K + u[:, None] + v[None, :]), 0)
            threshold = (tmp - 1).pow(2).sum().item()
            profile.errors.append(threshold)
            if threshold < config.sinkhorn_threshold:
                break

    output = torch.exp(K + u[:, None] + v[None, :])

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
) -> torch.Tensor:
    n, _ = C.shape

    start_time = TimeStamp(config.device)

    K = -C / config.sinkhorn_regularization

    v = -torch.logsumexp(K, 0)
    K = torch.exp(K + v[None, :])

    if cache is None or cache.u_guess is None:
        u = torch.full((n,), 1/n, device=config.device, dtype=config.dtype)
    else:
        u = cache.u_guess

    for iteration in range(config.sinkhorn_iterations):
        v = 1 / (u @ K + M_EPS)
        u = 1 / (K @ v + M_EPS)

        if iteration % config.sinkhorn_eval_freq == 0:
            b_hat = torch.matmul(u, K) * v
            threshold = (1 - b_hat).pow(2).sum().item()
            if threshold < config.sinkhorn_threshold:
                break

    output = u.reshape(-1, 1) * K * v.reshape(1, -1)

    if not cache is None:
        cache.update(u)

    profile.iteration_count = iteration + 1
    profile.time = TimeStamp(config.device).elapsed_seconds(start_time)

    return output
