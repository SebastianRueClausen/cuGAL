from functools import partial
import torch

from cugal.config import Config, SinkhornMethod

try:
    import cuda_kernels
    has_cuda = True
except ImportError:
    has_cuda = False

M_EPS = 1e-16


def can_use_cuda(config: Config) -> bool:
    return has_cuda and "cuda" in config.device and config.dtype in [
        torch.float32, torch.float16]


def sinkhorn(C: torch.Tensor, config: Config) -> torch.Tensor:
    match config.sinkhorn_method:
        case SinkhornMethod.STANDARD:
            return sinkhorn_knopp(C, config)[0]
        case SinkhornMethod.LOG:
            return loghorn(C, config)[0]
        case SinkhornMethod.MIX:
            return mixhorn(C, config)[0]


def sinkhorn_knopp(C: torch.Tensor, config: Config) -> tuple[torch.Tensor, int]:
    na, _ = C.shape

    u = torch.full(size=(na,), fill_value=1/na,
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


def loghorn(C: torch.Tensor, config: Config) -> tuple[torch.Tensor, int]:
    use_cuda = can_use_cuda(config)

    na, nb = C.shape

    K = - C / config.sinkhorn_regularization

    if use_cuda:
        K_transpose = K.t().contiguous()

    u = torch.zeros(na, device=config.device, dtype=config.dtype)
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
            if threshold < config.sinkhorn_threshold:
                break

    return torch.exp(K + u[:, None] + v[None, :]), iteration


def mixhorn(C: torch.Tensor, config: Config) -> tuple[torch.Tensor, int]:
    K = -C / config.sinkhorn_regularization

    v = -torch.logsumexp(K, 0)
    u = -torch.logsumexp(K + v[None, :], 1)

    K = torch.exp(K + u[:, None] + v[None, :])
    u = torch.exp(-u)

    prev_error = float("inf")

    for iteration in range(config.sinkhorn_iterations):
        v = 1 / (u @ K + M_EPS)
        u = 1 / (K @ v + M_EPS)

        if iteration % config.sinkhorn_eval_freq == 0:
            b_hat = torch.matmul(u, K) * v
            error = (1 - b_hat).pow(2).sum().item()
            if error < config.sinkhorn_threshold:
                break
            if abs(prev_error - error) < 1e-4:
                return loghorn(u.reshape(-1, 1) * K * v.reshape(1, -1), config)
            prev_error = error

    # u, v = newton(C, u, v)
    return u.reshape(-1, 1) * K * v.reshape(1, -1), iteration


def lyapunov(P: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    n = z.shape[0] // 2
    u, v = z[:n], z[n:]
    return -torch.sum(P) + torch.sum(u) + torch.sum(v)


def lyapunov_grad(P: torch.Tensor, _: torch.Tensor) -> torch.Tensor:
    return torch.cat([1 - torch.sum(P, 0), 1 - torch.sum(P, 1)], dim=0)


def lyapunov_hessian(P: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    row_sum = torch.sum(P, 0)
    col_sum = torch.sum(P, 1)

    row1 = torch.cat([torch.diag(row_sum), P], dim=1)
    row2 = torch.cat([P.T, torch.diag(col_sum)], dim=1)

    return torch.cat([row1, row2], dim=0)


def conjgrad(A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    x = torch.randn_like(b)
    r = b - A @ x
    p = r
    r_squared_old = torch.dot(r, r)
    for _ in b:
        Ap = A @ p
        alpha = r_squared_old / torch.dot(p, Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        r_squared_new = torch.dot(r, r)
        if torch.sqrt(r_squared_new) < 1e-8:
            break
        p = r + (r_squared_new/r_squared_old)*p
        r_squared_old = r_squared_new
    return x


def backtrack(func, grad_func, x, p, tau=0.5, alpha=1.0, c1=1e-3, max_iter=100):
    phi0, grad0 = func(x), grad_func(x)
    dphi0 = torch.dot(grad0, p)

    if dphi0 >= 0.0:
        return ValueError('Must provide a descent direction')

    for _ in range(max_iter):
        if torch.all(func(x + alpha*p) <= phi0 + c1*alpha*dphi0):
            return alpha
        alpha *= tau

    return None


def newton(C: torch.Tensor, u, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    n = len(u)

    for _ in range(2):
        P = torch.exp(-C + u[:, None] + v[None, :] - 1)
        M = lyapunov_hessian(P)
        z = torch.cat([u, v], dim=0)
        grad = conjgrad(M, -lyapunov_grad(P, z))
        alpha = backtrack(partial(lyapunov, P),
                          partial(lyapunov_grad, P), z, grad)
        assert not alpha is None
        u += alpha * grad[:n]
        v += alpha * grad[n:]

    return u, v
