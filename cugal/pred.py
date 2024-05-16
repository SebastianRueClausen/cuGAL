import networkx as nx
import numpy as np
import scipy
import scipy.optimize
import torch
from tqdm.auto import tqdm
from functools import partial

import lapjv

from cugal.adjacency import Adjacency
from cugal import sinkhorn
from cugal.config import Config, HungarianMethod
from cugal.profile import Profile, Phase, SinkhornProfile, TimeStamp
from cugal.feature_extraction import Features

from time import sleep

from cugal.hungarian_python import hungarian_algorithm
from cuda_hungarian import hungarian_torch

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
        cuda_kernels.regularize(gradient, P, iteration**2)
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

    if type(A) is Adjacency:
        P = torch.full([A.number_of_nodes()] * 2, fill_value=1 /
                        A.number_of_nodes(), device=config.device, dtype=config.dtype)
    else:
        P = torch.full([A.shape[0]] * 2, fill_value=1 /
                        A.shape[0], device=config.device, dtype=config.dtype)

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
                    del diff
                    break
            del diff

    return P

def hungarian(
    quasi_permutation: torch.Tensor,
    config: Config,
    profile: Profile,
) -> np.ndarray:
    match config.hungarian_method:
        case HungarianMethod.SCIPY:
            return hungarian_scipy(quasi_permutation, config, profile)
        case HungarianMethod.GREEDY:
            return hungarian_torch_python(quasi_permutation, config, profile)
        case HungarianMethod.RAND | HungarianMethod.MORE_RAND | HungarianMethod.DOUBLE_GREEDY | HungarianMethod.ENTRO_GREEDY | HungarianMethod.PARALLEL_GREEDY:
            return hungarian_cuda(quasi_permutation, config, profile)
        case HungarianMethod.JV:
            return jv_hungarian(quasi_permutation, config, profile)
        case _:
            raise NotImplementedError(f"Unsupported Hungarian method: {config.hungarian_method}")
        
def hungarian_torch_python(
    quasi_permutation: torch.Tensor,
    config: Config,
    profile: Profile,
) -> np.ndarray:
    quasi_permutation = quasi_permutation.cpu()
    quasi_permutation *= -1
    start_time = TimeStamp(config.device)
    col_ind = torch.empty(quasi_permutation.size(0), dtype=torch.int32, device='cpu')
    hungarian_torch(quasi_permutation, col_ind)
    profile.log_time(start_time, Phase.HUNGARIAN)
    return col_ind.tolist()

def hungarian_scipy(
    quasi_permutation: torch.Tensor,
    config: Config,
    profile: Profile,
) -> np.ndarray:
    quasi_permutation = quasi_permutation.cpu()
    start_time = TimeStamp(config.device)
    _, col_ind = scipy.optimize.linear_sum_assignment(
        quasi_permutation.clone(), maximize=True)
    profile.log_time(start_time, Phase.HUNGARIAN)
    return col_ind

def hungarian_cuda(
    quasi_permutation: torch.Tensor,
    config: Config,
    profile: Profile,
) -> np.ndarray:
    start_time = TimeStamp(config.device)
    col_ind = hungarian_algorithm(quasi_permutation, config)
    profile.log_time(start_time, Phase.HUNGARIAN)
    return col_ind

def jv_hungarian(
    quasi_permutation: torch.Tensor,
    config: Config,
    profile: Profile,
) -> np.ndarray:
    start_time = TimeStamp(config.device)
    col_ind, _, _ = lapjv.lapjv((quasi_permutation * -1).cpu().numpy())
    profile.log_time(start_time, Phase.HUNGARIAN)
    return col_ind    

def convert_to_permutation_matrix(
    quasi_permutation: torch.Tensor,
    config: Config,
    profile: Profile,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Convert quasi permutation matrix M into true permutation matrix.
    Also returns a mapping from source to target graph.
    """

    n = len(quasi_permutation)

    col_ind = hungarian(quasi_permutation, config, profile)

    permutation = np.zeros((n, n))
    mapping = []

    for i in range(n):
        permutation[i][col_ind[i]] = 1
        mapping.append((i, col_ind[i]))

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

