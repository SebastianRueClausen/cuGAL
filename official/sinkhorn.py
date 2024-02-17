# https://pythonot.github.io/_modules/ot/bregman.html#sinkhorn

import torch
from official.config import Config, SinkhornMethod

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
            return sinkhorn_log(a, b, C, config)

def sinkhorn_knopp(
    a: torch.Tensor,
    b: torch.Tensor,
    C: torch.Tensor,
    config: Config,
) -> torch.Tensor:
    na, nb = C.shape

    u = torch.full(size=(na,), fill_value=1/na, device=config.device, dtype=config.data_type)
    v = torch.full(size=(nb,), fill_value=1/nb, device=config.device, dtype=config.data_type)

    K = torch.empty(C.shape, device=config.device, dtype=config.data_type)
    torch.div(C, -config.sinkhorn_regularization, out=K)
    torch.exp(K, out=K)

    # allocate memory beforehand
    KTu = torch.empty(v.shape, device=config.device, dtype=config.data_type)
    Kv = torch.empty(u.shape, device=config.device, dtype=config.data_type)

    for _ in range(config.sinkhorn_iterations):
        torch.matmul(u, K, out=KTu)
        v = torch.div(b, KTu + M_EPS)
        torch.matmul(K, v, out=Kv)
        u = torch.div(a, Kv + M_EPS)

        #if torch.any(torch.isnan(u)) or torch.any(torch.isnan(v)) or \
        #        torch.any(torch.isinf(u)) or torch.any(torch.isinf(v)):
        #    print('Warning: numerical errors at iteration', it)
        #    u, v = upre, vpre
        #    break

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

    for _ in range(config.sinkhorn_iterations):
        v = logb - torch.logsumexp(K + u[:, None], 0)
        u = loga - torch.logsumexp(K + v[None, :], 1)

        #if iter % 50 == 0:
            #tmp = torch.sum(torch.exp(get_logT(Mr, u, v)), 0)
            #threshold = torch.norm(tmp - b)
            #if threshold < 0.01: break

    return torch.exp(get_logT(K, u, v))