import numpy as np
import torch
from tqdm.auto import tqdm
import networkx as nx
import scipy
from sklearn.metrics.pairwise import euclidean_distances
import cugal.sinkhorn as sinkhorn
from cugal.config import Config


def feature_extraction(G: nx.graph) -> np.ndarray:
    node_list = sorted(G.nodes())
    node_degree_dict = dict(G.degree())
    node_clustering_dict = dict(nx.clustering(G))
    egonets = {n: nx.ego_graph(G, n) for n in node_list}

    degs = [node_degree_dict[n] for n in node_list]
    clusts = [node_clustering_dict[n] for n in node_list]
    neighbor_degs = [
        np.mean([node_degree_dict[m] for m in egonets[n].nodes if m != n])
        if node_degree_dict[n] > 0
        else 0
        for n in node_list
    ]
    neighbor_clusts = [
        np.mean([node_clustering_dict[m] for m in egonets[n].nodes if m != n])
        if node_degree_dict[n] > 0
        else 0
        for n in node_list
    ]

    return np.nan_to_num(np.stack((degs, clusts, neighbor_degs, neighbor_clusts)).T)


def find_quasi_perm(A: np.ndarray, B: np.ndarray, D: np.ndarray, config: Config) -> torch.Tensor:
    n = len(A)

    A = torch.tensor(A, dtype=config.dtype, device=config.device)
    B = torch.tensor(B, dtype=config.dtype, device=config.device)
    D = torch.tensor(D, dtype=config.dtype, device=config.device)

    P = torch.full(size=(n, n), fill_value=1/n,
                   dtype=config.dtype, device=config.device)
    mat_ones = torch.ones((n, n), dtype=config.dtype, device=config.device)

    K = config.mu * D

    for i in tqdm(range(config.iter_count)):
        for it in range(1, 11):
            G = -A.T @ P @ B - A @ P @ B.T + K + i*(mat_ones - 2*P)
            q = sinkhorn.sinkhorn(G, config)
            alpha = 2.0 / float(2.0 + it)
            P = P + alpha * (q - P)

    return P.cpu()


def convert_to_perm_hungarian(
    M: torch.Tensor,
    n1: int,
    n2: int,
) -> list[tuple[float, float]]:
    n = len(M)

    row_ind, col_ind = scipy.optimize.linear_sum_assignment(M, maximize=True)

    P = np.zeros((n, n))
    ans = []

    for i in range(n):
        P[row_ind[i]][col_ind[i]] = 1
        if (row_ind[i] >= n1) or (col_ind[i] >= n2):
            continue
        ans.append((row_ind[i], col_ind[i]))

    return ans


def cugal(source: nx.graph, target: nx.graph, config: Config):
    """Run cuGAL algorithm."""

    node_count = source.number_of_nodes()

    if config.dtype == torch.float16 and source.number_of_nodes() % 2 != 0:
        source.add_node(source.number_of_nodes())
        target.add_node(target.number_of_nodes())

    F1 = feature_extraction(source)
    F2 = feature_extraction(target)

    P = find_quasi_perm(
        nx.to_numpy_array(source),
        nx.to_numpy_array(target),
        euclidean_distances(F1, F2),
        config,
    )

    mapping = convert_to_perm_hungarian(
        P, len(source.nodes()), len(target.nodes()))

    if node_count != len(mapping):
        mapping.pop()

    return mapping
