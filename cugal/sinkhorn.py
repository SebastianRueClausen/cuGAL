# https://pythonot.github.io/_modules/ot/bregman.html#sinkhorn

from torch.utils.cpp_extension import load_inline
import torch
from cugal.config import Config, SinkhornMethod
import numpy as np
import time
from functools import partial

try:
    import sinkhorn_cuda
    use_cuda = True
except ImportError:
    use_cuda = False

M_EPS = 1e-16


def sinkhorn(C: torch.Tensor, config: Config) -> torch.Tensor:
    match config.sinkhorn_method:
        case SinkhornMethod.STANDARD:
            return sinkhorn_knopp(C, config)[0]
        case SinkhornMethod.LOG if use_cuda:
            return sinkhorn_log_cuda(C, config)[0]
        case SinkhornMethod.LOG:
            return sinkhorn_log(C, config)[0]


def sinkhorn_knopp(C: torch.Tensor, config: Config) -> tuple[torch.Tensor, int]:
    na, nb = C.shape

    u = torch.full(size=(na,), fill_value=1/na,
                   device=config.device, dtype=config.dtype)
    v = torch.full(size=(nb,), fill_value=1/nb,
                   device=config.device, dtype=config.dtype)

    K = torch.exp(C / -config.sinkhorn_regularization)

    for iter in range(config.sinkhorn_iterations):
        v = 1 / (u @ K + M_EPS)
        u = 1 / (K @ v + M_EPS)

        if iter % config.sinkhorn_eval_freq == 0:
            b_hat = torch.matmul(u, K) * v
            threshold = (1 - b_hat).pow(2).sum().item()
            if threshold < config.sinkhorn_threshold:
                break

    return u.reshape(-1, 1) * K * v.reshape(1, -1), iter


def sinkhorn_log(C: torch.Tensor, config: Config) -> tuple[torch.Tensor, int]:
    na, nb = C.shape

    def get_logT(K, u, v):
        return K + u[:, None] + v[None, :]

    K = - C / config.sinkhorn_regularization

    u = torch.zeros(na, device=config.device, dtype=config.dtype)
    v = torch.zeros(nb, device=config.device, dtype=config.dtype)

    for iter in range(config.sinkhorn_iterations):
        v = -torch.logsumexp(K + u[:, None], 0)
        u = -torch.logsumexp(K + v[None, :], 1)

        if iter % config.sinkhorn_eval_freq == 0:
            tmp = torch.sum(torch.exp(get_logT(K, u, v)), 0)
            threshold = (tmp - 1).pow(2).sum().item()
            if threshold < config.sinkhorn_threshold:
                break

    return torch.exp(get_logT(K, u, v)), iter


def sinkhorn_log_cuda(C: torch.Tensor, config: Config) -> tuple[torch.Tensor, int]:
    na, nb = C.shape

    def get_logT(K, u, v):
        return K + u[:, None] + v[None, :]

    K = -C / config.sinkhorn_regularization
    K_transpose = K.t().contiguous()

    u = torch.zeros(na, device=config.device, dtype=config.dtype)
    v = torch.zeros(nb, device=config.device, dtype=config.dtype)

    for iter in range(config.sinkhorn_iterations):
        sinkhorn_cuda.sinkhorn_step(K_transpose, u, v)
        sinkhorn_cuda.sinkhorn_step(K, v, u)

        if iter % config.sinkhorn_eval_freq == 0:
            tmp = torch.sum(torch.exp(get_logT(K, u, v)), 0)
            threshold = (tmp - 1).pow(2).sum().item()
            if threshold < config.sinkhorn_threshold:
                break

    return torch.exp(get_logT(K, u, v)), iter


def mean_cuda_time(sinkhorn, iter_count: int) -> tuple[float, int]:
    times = []
    mean_required_iterations = 0

    for _ in range(iter_count):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        _, required_iterations = sinkhorn()
        mean_required_iterations += required_iterations
        end.record()

        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    return np.mean(times), mean_required_iterations // iter_count


def mean_cpu_time(sinkhorn, iter_count: int) -> tuple[float, int]:
    times = []
    mean_required_iterations = 0

    for _ in range(iter_count):
        before = time.time()
        _, required_iterations = sinkhorn()
        mean_required_iterations += required_iterations
        times.append(time.time() - before)

    return np.mean(times) * 1000, mean_required_iterations // iter_count


def benchmark_cuda():
    matrix_size = 10000
    iter_count = 3

    config = Config(device="cuda", dtype=torch.float32,
                    sinkhorn_iterations=200, sinkhorn_threshold=1e-8)
    cpu_config = Config(sinkhorn_iterations=200)

    matrix = torch.randn((matrix_size, matrix_size),
                         device=config.device, dtype=config.dtype) * 20.0

    mean_time_torch, mean_iter_torch = mean_cuda_time(
        partial(sinkhorn_log, matrix, config),
        iter_count,
    )

    mean_time_cuda, mean_iter_cuda = mean_cuda_time(
        partial(sinkhorn_log_cuda, matrix, config),
        iter_count,
    )

    mean_time_cpu, mean_iter_cpu = mean_cpu_time(
        partial(sinkhorn_knopp, matrix.cpu().to(
            cpu_config.dtype), cpu_config),
        iter_count,
    )

    print("torch - time:", mean_time_torch, "iter:", mean_iter_torch)
    print("cuda  - time:", mean_time_cuda, "iter", mean_iter_cuda)
    print("cpu   - time:", mean_time_cpu, "iter", mean_iter_cpu)
    print("cuda is time", mean_time_torch /
          mean_time_cuda, "times as fast as torch")
    print("cuda is time", mean_time_cpu /
          mean_time_cuda, "times as fast as cpu")
