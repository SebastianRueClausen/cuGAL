import torch
from cugal.config import Config, SinkhornMethod
from cugal.profile import SinkhornProfile, TimeStamp
from dataclasses import dataclass
import math

try:
    import cuda_kernels
    has_cuda = True
except ImportError:
    has_cuda = False

M_EPS = 1e-16


def can_use_cuda(config: Config) -> bool:
    return has_cuda and "cuda" in config.device and config.dtype == torch.float32


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
    return torch.mean(abs(torch.sum(scale_kernel_matrix(K.clone(), u, v), dim=0) - 1)).item()


def marginal_error_log(K: torch.Tensor, log_u: torch.Tensor, log_v: torch.Tensor) -> float:
    if has_cuda:
        return cuda_kernels.sinkhorn_log_marginal(K, log_u, log_v) / log_u.shape[0]
    return torch.mean(abs(torch.sum(scale_kernel_matrix_log(K.clone(), log_u, log_v), dim=0) - 1)).item()


@dataclass
class SinkhornState:
    u: torch.Tensor
    v: torch.Tensor

    def __init__(self, n: int, config: Config):
        match config.sinkhorn_method:
            case SinkhornMethod.LOG:
                vector = torch.zeros(
                    n, device=config.device, dtype=config.dtype)
            case SinkhornMethod.MIX | SinkhornMethod.STANDARD:
                vector = torch.full(size=(n,), fill_value=1/n,
                                    device=config.device, dtype=config.dtype)
        self.u = vector.clone()
        self.v = vector

    def marginal_error_log(self, K: torch.Tensor) -> float:
        return torch.sum(abs(torch.sum(scale_kernel_matrix_log(K.clone(), self.u, self.v), dim=0) - 1)).item()

    def lehmann_momentum(self, config: Config, errors: list[float]) -> float:
        if config.sinkhorn_momentum_start is None or len(errors) * config.sinkhorn_eval_freq < config.sinkhorn_momentum_start:
            return 1.0
        index = config.sinkhorn_momentum_start // config.sinkhorn_eval_freq
        if len(errors) < index - 1:
            return 1.0
        ratio = min(errors[index - 1] / errors[index - 2], 0.99)
        if math.isnan(ratio):
            ratio = 0.99
        power = 1.0 / config.sinkhorn_eval_freq
        return 2.0 / (1.0 + math.sqrt(1 - ratio ** power))

    def solve_log(
        self,
        C: torch.Tensor,
        config: Config,
        profile=SinkhornProfile(),
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        use_cuda = can_use_cuda(config)
        start_time = TimeStamp(config.device)
        K = - C / config.sinkhorn_regularization
        if use_cuda:
            K_transpose = K.t().contiguous()
        errors, momentum = [], 1.0
        for iteration in range(config.sinkhorn_iterations):
            if use_cuda:
                cuda_kernels.sinkhorn_log_step(
                    K_transpose, self.u, self.v, momentum)
                cuda_kernels.sinkhorn_log_step(K, self.v, self.u, momentum)
            else:
                self.v = self.v * (1 - momentum) - momentum * \
                    torch.logsumexp(K + self.u[:, None], 0)
                self.u = self.u * (1 - momentum) - momentum * \
                    torch.logsumexp(K + self.v[None, :], 1)
            if iteration % config.sinkhorn_eval_freq == 0 and iteration != 0:
                errors.append(self.marginal_error_log(K))
                if errors[-1] < config.sinkhorn_threshold:
                    break
                momentum = self.lehmann_momentum(config, errors)
        profile.iteration_count = iteration + 1
        profile.time = TimeStamp(config.device).elapsed_seconds(start_time)
        profile.errors = errors
        return K, self.u, self.v

    def solve_standard(
        self,
        C: torch.Tensor,
        config: Config,
        profile=SinkhornProfile(),
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        start_time = TimeStamp(config.device)
        K = torch.exp(C / -config.sinkhorn_regularization)
        errors, momentum = [], 1
        for iteration in range(config.sinkhorn_iterations):
            self.v = self.v ** (1 - momentum) * \
                (1 / (self.u @ K + M_EPS)) ** momentum
            self.u = self.u ** (1 - momentum) * \
                (1 / (K @ self.v + M_EPS)) ** momentum
            if iteration % config.sinkhorn_eval_freq == 0:
                errors.append(marginal_error(K, self.u, self.v))
                if errors[-1] < config.sinkhorn_threshold:
                    break
                momentum = self.lehmann_momentum(config, errors)
        profile.iteration_count = iteration + 1
        profile.time = TimeStamp(config.device).elapsed_seconds(start_time)
        profile.errors = errors
        return K, self.u, self.v

    def solve(
        self,
        C: torch.Tensor,
        config: Config,
        profile=SinkhornProfile(),
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        match config.sinkhorn_method:
            case SinkhornMethod.LOG:
                return self.solve_log(C, config, profile)
            case SinkhornMethod.STANDARD:
                return self.solve_standard(C, config, profile)
