import networkx as nx
import numpy as np
import scipy
import torch
from tqdm.auto import tqdm
from functools import partial

from cugal.adjacency import Adjacency
from cugal import sinkhorn
from cugal.sinkhorn import SinkhornCache
from cugal.config import Config
from cugal.profile import Profile, Phase, SinkhornProfile, TimeStamp
from cugal.feature_extraction import feature_distance_matrix

try:
    import cuda_kernels
    has_cuda = True
except ImportError:
    has_cuda = False


def dense_gradient(A, B, P, K: torch.Tensor, iteration: int) -> torch.Tensor:
    return -A.T @ P @ B - A @ P @ B.T + K + iteration*(1 - 2*P)


def sparse_gradient(
    A, B: Adjacency,
    A_transpose, B_transpose: Adjacency,
    P, K: torch.Tensor,
    iteration: int,
) -> torch.Tensor:
    if A is A_transpose and B is B_transpose:
        if has_cuda and "cuda" in str(K.device):
            out = torch.empty_like(K)
            cuda_kernels.calculate_gradient_symmetric(
                A.col_indices,
                A.row_pointers,
                B.col_indices,
                B.row_pointers,
                P, K, out, iteration,
            )
            return out
        else:
            return -2 * B.mul(A.mul(P).T).T + K + (iteration - iteration*2*P)
    else:
        return B_transpose.mul(A_transpose.mul(P, negate_lhs=True).T).T \
            - B.mul(A.mul(P).T).T + K + (iteration - iteration*2*P)


def find_quasi_permutation_matrix(
    A, B: nx.Graph | Adjacency,
    distance: torch.Tensor,
    config: Config,
    profile: Profile,
) -> torch.Tensor:
    if config.use_sparse_adjacency:
        if not type(A) is Adjacency:
            assert not nx.is_directed(
                A), "graph must be undirected to use sparse adjacency (for now)"
            A = Adjacency.from_graph(A, config.device)

        if not type(B) is Adjacency:
            assert not nx.is_directed(
                B), "graph must be undirected to use sparse adjacency (for now)"
            B = Adjacency.from_graph(B, config.device)
    else:
        A = torch.tensor(nx.to_numpy_array(
            A), dtype=config.dtype, device=config.device)
        B = torch.tensor(nx.to_numpy_array(
            B), dtype=config.dtype, device=config.device)

    P = torch.full_like(distance, fill_value=1/len(distance))
    K = config.mu * distance

    for λ in tqdm(range(config.iter_count), desc="λ"):
        if config.use_sinkhorn_cache:
            sinkhorn_cache = SinkhornCache()

        for it in tqdm(range(1, 11), desc="frank-wolfe", leave=False):
            start_time = TimeStamp(config.device)
            gradient_function = partial(sparse_gradient, A, B, A, B) \
                if config.use_sparse_adjacency else partial(dense_gradient, A, B)
            gradient = gradient_function(P, K, λ)
            profile.log_time(start_time, Phase.GRADIENT)

            start_time = TimeStamp(config.device)
            sinkhorn_profile = SinkhornProfile()

            sinkhorn_call = partial(
                sinkhorn.sinkhorn, gradient, config, sinkhorn_profile)
            q = sinkhorn_call(
                sinkhorn_cache) if config.use_sinkhorn_cache else sinkhorn_call()

            profile.sinkhorn_profiles.append(sinkhorn_profile)
            profile.log_time(start_time, Phase.SINKHORN)

            alpha = 2.0 / float(2.0 + it)
            P = P + alpha * (q - P)

    return P.cpu()


def convert_to_permutation_matrix(
    quasi_permutation: torch.Tensor,
    source_node_count: int,
    target_node_count: int,
    config: Config,
    profile: Profile,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Convert quasi permutation matrix M into true permutation matrix.
    Also returns a mapping from source to target graph.
    """

    n = len(quasi_permutation)

    start_time = TimeStamp(config.device)
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(
        quasi_permutation, maximize=True)
    profile.log_time(start_time, Phase.HUNGARIAN)

    permutation = np.zeros((n, n))
    mapping = []

    for i in range(n):
        permutation[row_ind[i]][col_ind[i]] = 1
        # if row_ind[i] >= source_node_count or col_ind[i] >= target_node_count:
        #    continue
        mapping.append((row_ind[i], col_ind[i]))

    return permutation, mapping


def max_node(graph: nx.Graph) -> int:
    return max(graph.nodes())


def cugal(
    source: nx.Graph,
    target: nx.Graph,
    config: Config,
    profile=Profile(),
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Run cuGAL algorithm.

    Returns permutation matrix and mapping from source to target.
    """

    source_node_count = max(source.nodes())
    target_node_count = max(target.nodes())
    node_count = max(source_node_count, target_node_count)

    # For float16 to be aligned properly in cuda, we add an extra dummy node
    # with no edges.
    if config.dtype == torch.float16 and node_count % 2 != 0:
        node_count += 1

    source.add_nodes_from(set(range(node_count)) - set(source.nodes()))
    target.add_nodes_from(set(range(node_count)) - set(target.nodes()))

    if config.use_sparse_adjacency and "cuda" in config.device:
        source, target = Adjacency.from_graph(
            source, config.device), Adjacency.from_graph(target, config.device)

    start_time = TimeStamp(config.device)
    distance = feature_distance_matrix(source, target, config)
    profile.log_time(start_time, Phase.FEATURE_EXTRACTION)

    quasi_permutation = find_quasi_permutation_matrix(
        source, target, distance, config, profile)

    return convert_to_permutation_matrix(
        quasi_permutation, source_node_count, target_node_count, config, profile)
