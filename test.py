import numpy as np
import official.pred as p
from official.config import Config, SinkhornMethod
import networkx as nx
import official.metrics as metrics
import torch, torch.cuda, torch.backends.mps
import matplotlib.pyplot as plt
import noise

def select_device() -> str:
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda:0'
    elif torch.backends.mps.is_available():
        device = 'mps'
    return device

cpu_config = Config(
    device='cpu',
    sinkhorn_regularization=1.0,
    sinkhorn_method=SinkhornMethod.STANDARD,
    sinkhorn_iterations=200,
    data_type=torch.float64,
    mu=1.0,
    iter_count=15,
)

gpu_log_config = Config(
    device=select_device(),
    sinkhorn_regularization=1.0,
    sinkhorn_method=SinkhornMethod.LOG,
    sinkhorn_iterations=200,
    data_type=torch.float16,
    mu=0.5,
    iter_count=15,
)

def permute_graph(
    source: nx.Graph,
    generator: np.random.Generator,
) -> tuple[nx.Graph, np.ndarray]:
    n = source.number_of_nodes()

    permutation = np.array((
        np.arange(n),
        generator.permutation(n)
    ))

    permutation = (
        permutation[:, permutation[1].argsort()][0],
        permutation[:, permutation[0].argsort()][1]
    )

    edges = np.array(source.edges)
    target = nx.Graph(permutation[0][edges].tolist())

    return target, permutation[1]

def test_graph_with_synthetic_noise(
    source: nx.Graph,
    config: Config,
    source_noise: float,
    target_noise: float,
):
    generator = np.random.default_rng()
    target, permutation = permute_graph(source, generator)

    edge_count = source.number_of_edges()

    source = noise.remove_edges(source, source_noise, generator)
    target = noise.remove_edges(target, target_noise, generator)

    source = noise.add_edges(source, edge_count - source.number_of_edges(), generator)
    target = noise.add_edges(target, edge_count - target.number_of_edges(), generator)

    test(source, target, permutation, config)

def test(G1: nx.Graph, G2: nx.Graph, correct_mapping: np.ndarray, config: Config):
    mapping = p.fugal(G1, G2, config)
    mapping = [x for _, x in mapping]

    assert len(G1.nodes()) == len(G2.nodes())

    A1 = nx.to_numpy_array(G1)
    A2 = nx.to_numpy_array(G2)

    assert A1.shape == A2.shape

    ics = metrics.induced_conserved_structure(A1, A2, correct_mapping, mapping)
    ec = metrics.edge_correctness(A1, A2, correct_mapping, mapping)
    sss = metrics.symmetric_substructure(A1, A2, correct_mapping, mapping)

    accuracy = np.mean(correct_mapping == mapping)

    print("induced conserved structure (ICS):", ics)
    print("edge correctness (EC):", ec)
    print("symmetric substructure score (SSS):", sss)
    print("accuracy:", accuracy)

def test_yeast(config: Config):
    f1 = open("./official/data/yeast0_Y2H1.txt", "r")
    f2 = open("./official/data/yeast10_Y2H1.txt", "r")

    nodes1 = [tuple(map(int, n.split())) for n in f1.read().split("\n") if n]
    nodes2 = [tuple(map(int, n.split())) for n in f2.read().split("\n") if n]

    n1 = max([max(n) for n in nodes1])
    n2 = max([max(n) for n in nodes2])

    assert n1 == n2

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

    generator = np.random.default_rng()
    G1 = noise.remove_edges(G2, 0.01, generator)

    n = max(len(G1.nodes()), len(G2.nodes()))
    for i in set(range(n)) - set(G1.nodes()):
        G1.add_node(i)

    for i in set(range(n)) - set(G2.nodes()):
        G2.add_node(i)

    assert len(G1.nodes()) == len(G2.nodes())

    test(G2, G1, np.arange(n1), config)

if __name__ == "__main__":
    test_yeast(cpu_config)