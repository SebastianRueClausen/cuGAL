# https://pythonot.github.io/_modules/ot/bregman.html#sinkhorn

from torch.utils.cpp_extension import load_inline
import torch
from impl.config import Config, SinkhornMethod
import numpy as np
import time
from functools import partial

with open("impl/sinkhorn.cpp") as file: cpp = file.read()
with open("impl/sinkhorn.cu") as file: cu = file.read()
module = load_inline(name="inline_extension", cpp_sources=[cpp], cuda_sources=[cu], functions=["sinkhorn_step"])

M_EPS = 1e-16

def sinkhorn(
    a: torch.Tensor,
    b: torch.Tensor,
    C: torch.Tensor,
    config: Config,
) -> torch.Tensor:
    match config.sinkhorn_method:
        case SinkhornMethod.STANDARD:
            return sinkhorn_knopp(a, b, C, config)[0]
        case SinkhornMethod.LOG if "cuda" in config.device:
            return sinkhorn_log_cuda(a, b, C, config)[0]
        case SinkhornMethod.LOG:
            return sinkhorn_log(a, b, C, config)[0]


def sinkhorn_knopp(
    a: torch.Tensor,
    b: torch.Tensor,
    C: torch.Tensor,
    config: Config,
) -> tuple[torch.Tensor, int]:
    na, nb = C.shape

    u = torch.full(size=(na,), fill_value=1/na,
                   device=config.device, dtype=config.data_type)
    v = torch.full(size=(nb,), fill_value=1/nb,
                   device=config.device, dtype=config.data_type)

    K = torch.exp(C / -config.sinkhorn_regularization)

    for iter in range(config.sinkhorn_iterations):
        v = b / (u @ K + M_EPS)
        u = a / (K @ v + M_EPS)

        if iter % config.sinkhorn_eval_freq == 0:
            b_hat = torch.matmul(u, K) * v
            threshold = (b - b_hat).pow(2).sum().item()
            if threshold < config.sinkhorn_threshold:
                break

    return u.reshape(-1, 1) * K * v.reshape(1, -1), iter


def sinkhorn_log(
    a: torch.Tensor,
    b: torch.Tensor,
    C: torch.Tensor,
    config: Config,
) -> tuple[torch.Tensor, int]:
    na, nb = len(a), len(b)

    def get_logT(K, u, v):
        return K + u[:, None] + v[None, :]

    K = - C / config.sinkhorn_regularization

    u = torch.zeros(na, device=config.device, dtype=config.data_type)
    v = torch.zeros(nb, device=config.device, dtype=config.data_type)

    loga, logb = torch.log(a), torch.log(b)

    for iter in range(config.sinkhorn_iterations):
        v = logb - torch.logsumexp(K + u[:, None], 0)
        u = loga - torch.logsumexp(K + v[None, :], 1)

        if iter % config.sinkhorn_eval_freq == 0:
            tmp = torch.sum(torch.exp(get_logT(K, u, v)), 0)
            threshold = (tmp - b).pow(2).sum().item()
            if threshold < config.sinkhorn_threshold:
                break

    return torch.exp(get_logT(K, u, v)), iter


# a and b must be ones.
def sinkhorn_log_cuda(
    a: torch.Tensor,
    b: torch.Tensor,
    C: torch.Tensor,
    config: Config,
) -> tuple[torch.Tensor, int]:
    na, nb = len(a), len(b)

    def get_logT(K, u, v):
        return K + u[:, None] + v[None, :]

    K = -C / config.sinkhorn_regularization
    K_transpose = K.t().contiguous()

    u = torch.zeros(na, device=config.device, dtype=config.data_type)
    v = torch.zeros(nb, device=config.device, dtype=config.data_type)

    for iter in range(config.sinkhorn_iterations):
        module.sinkhorn_step(K_transpose, u, v)
        module.sinkhorn_step(K, v, u)

        if iter % config.sinkhorn_eval_freq == 0:
            tmp = torch.sum(torch.exp(get_logT(K, u, v)), 0)
            threshold = (tmp - b).pow(2).sum().item()
            if threshold < config.sinkhorn_threshold:
                break

    return torch.exp(get_logT(K, u, v)), iter

def test_cuda():
    matrix_size = 1024 * 10
    dtype = torch.float16

    matrix = torch.randn((matrix_size, matrix_size), device="cuda", dtype=dtype) * 42.0
    matrix_transposed = matrix.t().contiguous()

    vector = torch.randn(matrix_size, device="cuda", dtype=dtype)

    a0 = -torch.logsumexp(matrix + vector[:, None], 0)
    a1 = -torch.logsumexp(matrix + vector[None, :], 1)

    b0 = torch.empty_like(a0, device="cuda")
    b1 = torch.empty_like(a0, device="cuda")
    module.sinkhorn_step(matrix_transposed, vector, b0)
    module.sinkhorn_step(matrix, vector, b1)

    d0 = torch.mean(torch.abs(a0 - b0)).item()
    d1 = torch.mean(torch.abs(a1 - b1)).item()

    print("mean difference axis 0:", d0)
    print("mean difference axis 1:", d1)

    # assert torch.allclose(a0, b0)
    # assert torch.allclose(a1, b1)

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

    config = Config(device="cuda", data_type=torch.float32, sinkhorn_iterations=200, sinkhorn_threshold=1e-8)
    cpu_config = Config(sinkhorn_iterations=200)

    matrix = torch.randn((matrix_size, matrix_size), device=config.device, dtype=config.data_type) * 20.0
    ones = torch.ones((matrix_size,), device=config.device, dtype=config.data_type)

    matrix_cpu = matrix.cpu().to(cpu_config.data_type)
    ones_cpu = ones.cpu().to(cpu_config.data_type)

    mean_time_torch, mean_iter_torch = mean_cuda_time(
        partial(sinkhorn_log, ones, ones, matrix, config),
        iter_count,
    )

    mean_time_cuda, mean_iter_cuda = mean_cuda_time(
        partial(sinkhorn_log_cuda, ones, ones, matrix, config),
        iter_count,
    )

    mean_time_cpu, mean_iter_cpu = mean_cpu_time(
        partial(sinkhorn_knopp, ones_cpu, ones_cpu, matrix_cpu, cpu_config),
        iter_count,
    )

    print("torch - time:", mean_time_torch, "iter:", mean_iter_torch)
    print("cuda  - time:", mean_time_cuda, "iter", mean_iter_cuda)
    print("cpu   - time:", mean_time_cpu, "iter", mean_iter_cpu)
    print("cuda is time", mean_time_torch / mean_time_cuda, "times as fast as torch")
    print("cuda is time", mean_time_cpu / mean_time_cuda, "times as fast as cpu")

