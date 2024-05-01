import networkx as nx
import numpy as np
import scipy
import torch
from tqdm.auto import tqdm
from functools import partial

from cugal.adjacency import Adjacency
from cugal import sinkhorn
from cugal.config import Config
from cugal.profile import Profile, Phase, SinkhornProfile, TimeStamp
from cugal.feature_extraction import Features

try:
    import cuda_kernels
    has_cuda = True
except ImportError:
    has_cuda = False



def add_feature_distance(gradient: torch.Tensor, features: torch.Tensor | Features) -> torch.Tensor:
    if type(features) is Features:
        gradient = features.add_distance(gradient)
    else:
        gradient += features
    return gradient


def dense_gradient(
    A: torch.Tensor,
    B: torch.Tensor,
    P: torch.Tensor,
    features: torch.Tensor | Features,
    iteration: int,
) -> torch.Tensor:
    gradient = -A.T @ P @ B - A @ P @ B.T
    return add_feature_distance(gradient, features) + iteration*(1 - 2*P)


def sparse_gradient(
    A, B: Adjacency,
    A_transpose: Adjacency,
    B_transpose: Adjacency,
    P: torch.Tensor,
    features: torch.Tensor | Features,
    iteration: int,
) -> torch.Tensor:
    if A is A_transpose and B is B_transpose:
        gradient = B.mul(A.mul(P).T).T
        gradient *= -2
    else:
        gradient = B_transpose.mul(A_transpose.mul(P, negate_lhs=True).T).T \
            - B.mul(A.mul(P).T).T

    gradient = add_feature_distance(gradient, features)

    if has_cuda and 'cuda' in str(P.device):
        cuda_kernels.regularize(gradient, P, iteration)
    else:
        gradient += iteration - iteration * 2 * P
    
    return gradient


def find_quasi_permutation_matrix(
    A, B: nx.Graph | Adjacency,
    features: torch.Tensor | Features,
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

    sinkhorn_cache = sinkhorn.init_from_cache_size(config.sinkhorn_cache_size)

    P = torch.full([A.number_of_nodes()] * 2, fill_value=1 /
                   A.number_of_nodes(), device=config.device, dtype=config.dtype)

    for λ in tqdm(range(config.iter_count), desc="λ"):
        for it in tqdm(range(1, config.frank_wolfe_iter_count + 1), desc="frank-wolfe", leave=False):
            start_time = TimeStamp(config.device)
            gradient_function = partial(sparse_gradient, A, B, A, B) \
                if config.use_sparse_adjacency else partial(dense_gradient, A, B)
            gradient = gradient_function(P, features, λ)
            profile.log_time(start_time, Phase.GRADIENT)

            start_time = TimeStamp(config.device)
            sinkhorn_profile = SinkhornProfile()

            q = sinkhorn.sinkhorn(
                gradient, config, sinkhorn_profile, sinkhorn_cache)

            profile.sinkhorn_profiles.append(sinkhorn_profile)
            profile.log_time(start_time, Phase.SINKHORN)

            alpha = 2.0 / float(2.0 + it)
            q -= P
            q *= alpha
            diff = q
            del q

            P += diff

            if not config.frank_wolfe_threshold is None:
                if diff.max() < config.frank_wolfe_threshold:
                    break
            del diff

    return P.cpu()


def convert_to_permutation_matrix(
    quasi_permutation: torch.Tensor,
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

    before = TimeStamp('cpu')

    node_count = max(max(source.nodes()), max(target.nodes()))

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
    features = Features.create(source, target, config)

    if not config.recompute_distance:
        features = features.distance_matrix()

    profile.log_time(start_time, Phase.FEATURE_EXTRACTION)

    quasi_permutation = find_quasi_permutation_matrix(
        source, target, features, config, profile)
    output = convert_to_permutation_matrix(quasi_permutation, config, profile)

    profile.time = TimeStamp('cpu').elapsed_seconds(before)

    return output
