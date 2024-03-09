import torch

from cugal.config import Config, SinkhornMethod

try:
    import cuda_kernels
    has_cuda = True
except ImportError:
    has_cuda = False

M_EPS = 1e-16


def sinkhorn(C: torch.Tensor, config: Config) -> torch.Tensor:
    use_cuda = has_cuda and "cuda" in config.device and config.dtype in [
        torch.float32, torch.float16]
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

    for iteration in range(config.sinkhorn_iterations):
        v = 1 / (u @ K + M_EPS)
        u = 1 / (K @ v + M_EPS)

        if iteration % config.sinkhorn_eval_freq == 0:
            b_hat = torch.matmul(u, K) * v
            threshold = (1 - b_hat).pow(2).sum().item()
            if threshold < config.sinkhorn_threshold:
                break

    return u.reshape(-1, 1) * K * v.reshape(1, -1), iteration


def sinkhorn_log(C: torch.Tensor, config: Config) -> tuple[torch.Tensor, int]:
    na, nb = C.shape

    K = - C / config.sinkhorn_regularization

    u = torch.zeros(na, device=config.device, dtype=config.dtype)
    v = torch.zeros(nb, device=config.device, dtype=config.dtype)

    for iteration in range(config.sinkhorn_iterations):
        v = -torch.logsumexp(K + u[:, None], 0)
        u = -torch.logsumexp(K + v[None, :], 1)

        if iteration % config.sinkhorn_eval_freq == 0:
            tmp = torch.sum(torch.exp(K + u[:, None] + v[None, :]), 0)
            threshold = (tmp - 1).pow(2).sum().item()
            if threshold < config.sinkhorn_threshold:
                break

    return torch.exp(K + u[:, None] + v[None, :]), iteration


def sinkhorn_log_cuda(C: torch.Tensor, config: Config) -> tuple[torch.Tensor, int]:
    na, nb = C.shape

    K = -C / config.sinkhorn_regularization
    K_transpose = K.t().contiguous()

    u = torch.zeros(na, device=config.device, dtype=config.dtype)
    v = torch.zeros(nb, device=config.device, dtype=config.dtype)

    for iteration in range(config.sinkhorn_iterations):
        cuda_kernels.sinkhorn_step(K_transpose, u, v)
        cuda_kernels.sinkhorn_step(K, v, u)

        if iteration % config.sinkhorn_eval_freq == 0:
            tmp = torch.sum(torch.exp(K + u[:, None] + v[None, :]), 0)
            threshold = (tmp - 1).pow(2).sum().item()
            if threshold < config.sinkhorn_threshold:
                break

    return torch.exp(K + u[:, None] + v[None, :]), iteration
