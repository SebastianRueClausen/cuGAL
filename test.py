import numpy as np
import official.pred as p
from official.config import Config, SinkhornMethod
import networkx as nx
import official.metrics as metrics
import torch, torch.cuda, torch.backends.mps

def select_device() -> torch.device:
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    return torch.device(device, 0)

gpu_config = Config(
    device=select_device(),
    sinkhorn_regularization=8.0,
    sinkhorn_method=SinkhornMethod.STANDARD,
    sinkhorn_iterations=200,
    data_type=torch.float32,
    mu=1.0,
    iter_count=15,
)

cpu_config = Config(
    device='cpu',
    sinkhorn_regularization=1.0,
    sinkhorn_method=SinkhornMethod.LOG,
    sinkhorn_iterations=200,
    data_type=torch.float64,
    mu=1.0,
    iter_count=15,
)

gpu_fast_log_config = Config(
    device=select_device(),
    sinkhorn_regularization=8.0,
    sinkhorn_method=SinkhornMethod.LOG_FAST,
    sinkhorn_iterations=200,
    data_type=torch.float32,
    mu=1.0,
    iter_count=15,
)

gpu_slow_log_config = Config(
    device=select_device(),
    sinkhorn_regularization=1.0,
    sinkhorn_method=SinkhornMethod.LOG,
    sinkhorn_iterations=200,
    data_type=torch.float16,
    mu=1.0,
    iter_count=15,
)

def test_yeast():
    f1 = open("./official/data/yeast0_Y2H1.txt", "r")
    f2 = open("./official/data/yeast10_Y2H1.txt", "r")

    nodes1 = [list(map(int, n.split())) for n in f1.read().split("\n") if n]
    nodes2 = [list(map(int, n.split())) for n in f2.read().split("\n") if n]

    n1 = max([max(n) for n in nodes1])
    n2 = max([max(n) for n in nodes2])

    assert(n1 == n2)

    f1.close()
    f2.close()

    A1 = np.zeros((n1, n1))
    for i, j in nodes1:
        A1[i - 1, j - 1] = 1
        A1[j - 1, i - 1] = 1

    A2 = np.zeros((n1, n1))
    for i, j in nodes2:
        A2[i - 1, j - 1] = 1
        A2[j - 1, i - 1] = 1

    G1 = nx.from_numpy_array(A1)
    G2 = nx.from_numpy_array(A2)

    mapping = p.fugal(G1, G2, gpu_slow_log_config)
    mapping = [x for _, x in mapping]

    print("ICS:", metrics.ICS(A1, A2, np.arange(n1), mapping))

if __name__ == "__main__":
    test_yeast()