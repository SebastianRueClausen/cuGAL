import numpy as np
import torch
from tqdm.auto import tqdm
import networkx as nx
import scipy
from sklearn.metrics.pairwise import euclidean_distances
import official.sinkhorn as sinkhorn
from official.config import Config

def feature_extraction(G: nx.graph):
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

    A = torch.tensor(A, dtype=config.data_type, device=config.device)
    B = torch.tensor(B, dtype=config.data_type, device=config.device)
    D = torch.tensor(D, dtype=config.data_type, device=config.device)

    P = torch.full(size=(n,n), fill_value=1/n, dtype=config.data_type, device=config.device)
    ones = torch.ones(n, dtype=config.data_type, device=config.device)
    mat_ones = torch.ones((n, n), dtype=config.data_type, device=config.device)

    K = config.mu * D

    for i in tqdm(range(config.iter_count)):
        for it in range(1, 11):
            G = -torch.mm(torch.mm(A.T, P), B) - torch.mm(torch.mm(A, P), B.T) + K + i*(mat_ones - 2*P)
            q = sinkhorn.sinkhorn(ones, ones, G, config)
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

def fugal(Gq: nx.graph, Gt: nx.graph, config: Config):
    n1 = len(Gq.nodes())
    n2 = len(Gt.nodes())
    n = max(n1, n2)

    for i in range(n1, n): Gq.add_node(i)
    for i in range(n2, n): Gt.add_node(i)

    F1 = feature_extraction(Gq)
    F2 = feature_extraction(Gt)

    P = find_quasi_perm(
        nx.to_numpy_array(Gq),
        nx.to_numpy_array(Gt),
        euclidean_distances(F1, F2),
        config,
    )

    return convert_to_perm_hungarian(P, n1, n2)