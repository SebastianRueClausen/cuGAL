import networkx as nx
import numpy as np
import scipy
import torch
from sklearn.metrics.pairwise import euclidean_distances
from tqdm.auto import tqdm

from cugal import sinkhorn
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


def find_quasi_perm(
    A: np.ndarray,
    B: np.ndarray,
    D: np.ndarray,
    config: Config,
) -> torch.Tensor:
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


def convert_to_permutation_matrix(
    quasi_permutation: torch.Tensor,
    source_node_count: int,
    target_node_count: int,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Convert quasi permutation matrix M into true permutation matrix.
    Also returns a mapping from source to target graph.
    """
    n = len(quasi_permutation)

    row_ind, col_ind = scipy.optimize.linear_sum_assignment(
        quasi_permutation, maximize=True)

    permutation = np.zeros((n, n))
    mapping = []

    for i in range(n):
        permutation[row_ind[i]][col_ind[i]] = 1
        if (row_ind[i] >= source_node_count) or (col_ind[i] >= target_node_count):
            continue
        mapping.append((int(row_ind[i]), int(col_ind[i])))

    return permutation, mapping


def cugal(
    source: nx.graph,
    target: nx.graph,
    config: Config,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Run cuGAL algorithm.

    Returns permutation matrix and mapping from source to target.
    """
    source_node_count = source.number_of_nodes()
    target_node_count = target.number_of_nodes()

    node_count = max(source_node_count, target_node_count)

    # For float16 to be aligned properly in cuda, we add an extra dummy node
    # with no edges.
    if config.dtype == torch.float16 and node_count % 2 != 0:
        node_count += 1

    # Add dummy nodes to match sizes.
    while source.number_of_nodes() < node_count:
        source.add_node(source.number_of_nodes())

    while target.number_of_nodes() < node_count:
        target.add_node(target.number_of_nodes())

    source_features = feature_extraction(source)
    target_features = feature_extraction(target)

    quasi_permutation = find_quasi_perm(
        nx.to_numpy_array(source),
        nx.to_numpy_array(target),
        euclidean_distances(source_features, target_features),
        config,
    )

    return convert_to_permutation_matrix(
        quasi_permutation, source_node_count, target_node_count)
