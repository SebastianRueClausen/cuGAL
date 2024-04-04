import networkx as nx
import numpy as np
import scipy
import torch
from sklearn.metrics.pairwise import euclidean_distances
from tqdm.auto import tqdm
from functools import partial
from time import time

from cugal.adjacency import Adjacency
from cugal import sinkhorn
from cugal.config import Config
from cugal.profile import Profile, Phase, SinkhornProfile


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


def dense_gradient(A, B, P, K: torch.Tensor, iteration: int) -> torch.Tensor:
    return -A.T @ P @ B - A @ P @ B.T + K + iteration*(1 - 2*P)


def sparse_gradient(
    A, B: Adjacency,
    A_transpose, B_transpose: Adjacency,
    P, K: torch.Tensor,
    iteration: int,
) -> torch.Tensor:
    # TODO: Figure out why there are small numeric differences between the non-sparse.
    return B_transpose.mul(A_transpose.mul(P, negate_lhs=True).T).T \
        - B.mul(A.mul(P).T).T + K + iteration*(1 - 2*P)


def find_quasi_permutation_matrix(
    A: nx.graph,
    B: nx.graph,
    distance: np.ndarray,
    config: Config,
    profile: Profile,
) -> torch.Tensor:
    n = len(A)

    distance = torch.tensor(distance, dtype=config.dtype, device=config.device)

    if config.use_sparse_adjacency:
        assert not nx.is_directed(A), "graph must be undirected to use sparse adjacency (for now)"
        assert not nx.is_directed(B), "graph must be undirected to use sparse adjacency (for now)"
        A, B = Adjacency.from_graph(A, config.device), Adjacency.from_graph(B, config.device)
    else:
        A = torch.tensor(nx.to_numpy_array(A), dtype=config.dtype, device=config.device)
        B = torch.tensor(nx.to_numpy_array(B), dtype=config.dtype, device=config.device)

    P = torch.full(size=(n, n), fill_value=1/n,
                   dtype=config.dtype, device=config.device)

    K = config.mu * distance

    for i in tqdm(range(config.iter_count)):
        for it in range(1, 11):
            start_time = time()
            gradient_function = partial(sparse_gradient, A, B, A, B) \
                if config.use_sparse_adjacency else partial(dense_gradient, A, B)
            gradient = gradient_function(P, K, i)
            profile.log_time(start_time, Phase.GRADIENT) 

            start_time = time()
            sinkhorn_profile = SinkhornProfile()
            q = sinkhorn.sinkhorn(gradient, config, sinkhorn_profile)
            profile.sinkhorn_profiles.append(sinkhorn_profile)
            profile.log_time(start_time, Phase.SINKHORN)

            alpha = 2.0 / float(2.0 + it)
            P = P + alpha * (q - P)

    return P.cpu()


def convert_to_permutation_matrix(
    quasi_permutation: torch.Tensor,
    source_node_count: int,
    target_node_count: int,
    profile: Profile,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Convert quasi permutation matrix M into true permutation matrix.
    Also returns a mapping from source to target graph.
    """

    n = len(quasi_permutation)

    start_time = time()
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(
        quasi_permutation, maximize=True)
    profile.log_time(start_time, Phase.HUNGARIAN)

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
    profile = Profile(),
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

    start_time = time()
    source_features = feature_extraction(source)
    target_features = feature_extraction(target)
    profile.log_time(start_time, Phase.FEATURE_EXTRACTION)

    distance = euclidean_distances(source_features, target_features)
    quasi_permutation = find_quasi_permutation_matrix(source, target, distance, config, profile)

    return convert_to_permutation_matrix(
        quasi_permutation, source_node_count, target_node_count, profile)