# https://pythonot.github.io/_modules/ot/bregman.html#sinkhorn

from torch.utils.cpp_extension import load_inline
from os import read
import torch
from impl.config import Config, SinkhornMethod

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

with open("impl/sinkhorn.cpp") as file:
    cpp = file.read()

with open("impl/sinkhorn.cu") as file:
    cu = file.read()

module = load_inline(name="inline_extension", cpp_sources=[cpp], cuda_sources=[cu], functions=["sinkhorn_step_row", "sinkhorn_step_col"])

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

    u = torch.zeros(na, device=config.device, dtype=config.data_type)
    v = torch.zeros(nb, device=config.device, dtype=config.data_type)

    for iter in range(config.sinkhorn_iterations):
        v = module.sinkhorn_step_row(K, u)
        u = module.sinkhorn_step_col(K, v)

        if iter % config.sinkhorn_eval_freq == 0:
            tmp = torch.sum(torch.exp(get_logT(K, u, v)), 0)
            threshold = (tmp - b).pow(2).sum().item()
            if threshold < config.sinkhorn_threshold:
                break

    return torch.exp(get_logT(K, u, v))