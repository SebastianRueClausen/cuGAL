# https://pythonot.github.io/_modules/ot/bregman.html#sinkhorn

from torch.utils.cpp_extension import load_inline
import torch
from impl.config import Config, SinkhornMethod
import numpy as np

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
            return sinkhorn_knopp(a, b, C, config)
        case SinkhornMethod.LOG:
            return sinkhorn_log_cuda(a, b, C, config)


def sinkhorn_knopp(
    a: torch.Tensor,
    b: torch.Tensor,
    C: torch.Tensor,
    config: Config,
) -> torch.Tensor:
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

    return u.reshape(-1, 1) * K * v.reshape(1, -1)


def sinkhorn_log(
    a: torch.Tensor,
    b: torch.Tensor,
    C: torch.Tensor,
    config: Config,
) -> torch.Tensor:
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

    return torch.exp(get_logT(K, u, v))

# a and b must be ones.
def sinkhorn_log_cuda(
    a: torch.Tensor,
    b: torch.Tensor,
    C: torch.Tensor,
    config: Config,
) -> torch.Tensor:
    na, nb = len(a), len(b)

    def get_logT(K, u, v):
        return K + u[:, None] + v[None, :]

    K = -C / config.sinkhorn_regularization
    K_transpose = K.t().clone()

    u = torch.zeros(na, device=config.device, dtype=config.data_type)
    v = torch.zeros(nb, device=config.device, dtype=config.data_type)

    for iter in range(config.sinkhorn_iterations):
        v = module.sinkhorn_step(K, u, 0)
        u = module.sinkhorn_step(K_transpose, v, 0)

        if iter % config.sinkhorn_eval_freq == 0:
            tmp = torch.sum(torch.exp(get_logT(K, u, v)), 0)
            threshold = (tmp - b).pow(2).sum().item()
            if threshold < config.sinkhorn_threshold:
                break

    return torch.exp(get_logT(K, u, v))

def test_cuda():
    matrix_size = 1000
    dtype = torch.float32

    matrix = torch.randn((matrix_size, matrix_size), device="cuda", dtype=dtype)
    matrix_transposed = matrix.t().clone()

    vector = torch.randn(matrix_size, device="cuda", dtype=dtype)

    a0 = -torch.logsumexp(matrix + vector[:, None], 0)
    a1 = -torch.logsumexp(matrix + vector[None, :], 1)

    b0 = module.sinkhorn_step(matrix, vector, 0)
    b1 = module.sinkhorn_step(matrix_transposed, vector, 0)

    d0 = torch.mean(torch.abs(a0 - b0)).item()
    d1 = torch.mean(torch.abs(a1 - b1)).item()

    print("mean difference axis 0:", d0)
    print("mean difference axis 1:", d1)

    assert torch.allclose(a0, b0)
    assert torch.allclose(a1, b1)

def mean_cuda_time(function, iter_count) -> float:
    times = []

    for _ in range(iter_count):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        function()
        end.record()

        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    return np.mean(times)


def benchmark_cuda():
    matrix_size = 100
    iter_count = 15000
    dtype = torch.float32

    matrix = torch.randn((matrix_size, matrix_size), device="cuda", dtype=dtype)
    matrix_transposed = matrix.t().clone()
    vector = torch.randn(matrix_size, device="cuda", dtype=dtype)

    def test_torch():
        -torch.logsumexp(matrix + vector[:, None], 0)
        -torch.logsumexp(matrix + vector[None, :], 1)

    def test_cuda():
        module.sinkhorn_step(matrix, vector, 0)
        module.sinkhorn_step(matrix_transposed, vector, 0)

    mean_torch = mean_cuda_time(test_torch, iter_count)
    mean_cuda = mean_cuda_time(test_cuda, iter_count)

    print("torch time:", mean_torch)
    print("cuda time:", mean_cuda)
    print("cuda is time", mean_torch / mean_cuda, "times faster")
    