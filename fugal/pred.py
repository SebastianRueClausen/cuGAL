import numpy as np
import torch
from tqdm.auto import tqdm
import networkx as nx
import time
import scipy
from sklearn.metrics.pairwise import euclidean_distances
from fugal import sinkhorn
from cugal.profile import Profile, Phase, SinkhornProfile, TimeStamp
from cugal.config import Config


def feature_extraction(G):
    """Node feature extraction.

    Parameters
    ----------

    G (nx.Graph): a networkx graph.

    Returns
    -------

    node_features (float): the Nx7 matrix of node features."""

    # necessary data structures
    node_features = np.zeros(shape=(G.number_of_nodes(), 7))
    node_list = sorted(G.nodes())
    node_degree_dict = dict(G.degree())
    node_clustering_dict = dict(nx.clustering(G))
    egonets = {n: nx.ego_graph(G, n) for n in node_list}

    # node degrees
    degs = [node_degree_dict[n] for n in node_list]

    # clustering coefficient
    clusts = [node_clustering_dict[n] for n in node_list]

    # average degree of neighborhood
    neighbor_degs = [
        np.mean([node_degree_dict[m] for m in egonets[n].nodes if m != n])
        if node_degree_dict[n] > 0
        else 0
        for n in node_list
    ]

    # average clustering coefficient of neighborhood
    neighbor_clusts = [
        np.mean([node_clustering_dict[m] for m in egonets[n].nodes if m != n])
        if node_degree_dict[n] > 0
        else 0
        for n in node_list
    ]

    # assembling the features
    node_features[:, 0] = degs
    node_features[:, 1] = clusts
    node_features[:, 2] = neighbor_degs
    node_features[:, 3] = neighbor_clusts

    node_features = np.nan_to_num(node_features)
    return np.nan_to_num(node_features)


def eucledian_dist(F1, F2, n):
    D = euclidean_distances(F1, F2)
    return D


def dist(A, B, P):
    obj = np.linalg.norm(np.dot(A, P) - np.dot(P, B))
    return obj*obj/2


def FindQuasiPerm(A, B, D, mu, niter, profile, config: Config):
    n = len(A)
    P = torch.ones((n, n), dtype=torch.float64) / n
    ones = torch.ones(n, dtype=torch.float64)
    mat_ones = torch.ones((n, n), dtype=torch.float64)
    reg = 1.0
    K = mu * D
    for i in tqdm(range(niter), desc="λ"):
        for it in tqdm(range(1, 11), desc="frank-wolfe", leave=False):
            start_time = TimeStamp('cpu')
            G: torch.Tensor = -torch.mm(torch.mm(A.T, P), B) - \
                torch.mm(torch.mm(A, P), B.T) + K + i*(mat_ones - 2*P)
            if config.safe_mode:
                #Check for NaN values
                assert torch.isfinite(G).all(), "G tensor has NaN values"
            profile.log_time(start_time, Phase.GRADIENT)

            start_time = TimeStamp('cpu')
            #clamp_start_time = TimeStamp('cpu')
            #torch.clamp(G, min=-600, max=600, out=G)
            np.nan_to_num(G, copy=False)
            #profile.log_time(clamp_start_time, Phase.CLAMP)
            q = sinkhorn.sinkhorn(ones, ones, G, reg,
                                  maxIter=500, stopThr=1e-3)
            np.nan_to_num(q, copy=False)
            if config.safe_mode:
                #Check for NaN values
                assert torch.isfinite(q).all(), "q tensor has NaN values"
            profile.log_time(start_time, Phase.SINKHORN)

            alpha = 2.0 / float(2.0 + it)
            P = P + alpha * (q - P)

            if config.safe_mode:
                #Check for NaN values
                assert torch.isfinite(P).all(), "P tensor has NaN values"

    return P


def convertToPermHungarian(M, n1, n2):
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(M, maximize=True)
    n = len(M)

    P = np.zeros((n, n))
    ans = []
    for i in range(n):
        P[row_ind[i]][col_ind[i]] = 1
        if (row_ind[i] >= n1) or (col_ind[i] >= n2):
            continue
        ans.append((row_ind[i], col_ind[i]))
    return P, ans


def fugal(Gq, Gt, mu, niter, config: Config, profile: Profile):
    n1 = len(Gq.nodes())
    n2 = len(Gt.nodes())
    n = max(n1, n2)
    for i in range(n1, n):
        Gq.add_node(i)
    for i in range(n2, n):
        Gt.add_node(i)

    print("Graphs have been padded to size", n)
    A = torch.tensor((nx.to_numpy_array(Gq)), dtype=torch.float64)
    B = torch.tensor((nx.to_numpy_array(Gt)), dtype=torch.float64)

    before = TimeStamp('cpu')

    print("Feature extraction")
    F1 = feature_extraction(Gq)
    F2 = feature_extraction(Gt)
    if config.safe_mode:
        #Check for NaN values
        assert np.isfinite(F1).all(), "F1 tensor has NaN values"
        assert np.isfinite(F2).all(), "F2 tensor has NaN values"

    print("Calculating eucledian distance")
    D = eucledian_dist(F1, F2, n)
    D = torch.tensor(D, dtype=torch.float64)
    if config.safe_mode:
        #Check for NaN values
        assert torch.isfinite(D).all(), "D tensor has NaN values"

    profile.log_time(before, Phase.FEATURE_EXTRACTION)

    print("Finding quasi permutation")
    P = FindQuasiPerm(A, B, D, mu, niter, profile, config=config)

    if config.safe_mode:
        #Check for NaN values
        assert torch.isfinite(P).all(), "P tensor has NaN values"

    start_time = TimeStamp('cpu')
    print("Converting to permutation")
    P_perm, ans = convertToPermHungarian(P, n1, n2)
    profile.log_time(start_time, Phase.HUNGARIAN)
    return P_perm, ans


def predict_alignment(queries, targets, mu=1, niter=15):
    n = len(queries)
    mapping = []
    times = []
    for i in tqdm(range(n)):
        t1 = time.time()
        ans = fugal(queries[i], targets[i], mu, niter)
        mapping.append(ans)
        t2 = time.time()
        times.append(t2 - t1)
    return mapping, times
