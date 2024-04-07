import networkx as nx
import numpy as np
import cuda_kernels
import torch


def graph_clustering(graph: nx.graph) -> np.ndarray:
    clustering = nx.clustering(graph)
    return np.array([clustering[i] for i in graph.nodes()])


def graph_clustering_cuda(graph: nx.graph) -> np.ndarray:
    edges = list(nx.edges(graph))
    if not graph.is_directed():
        edges += [(y, x) for x, y in edges]
    edges_tensor = torch.stack(list(map(torch.tensor, sorted(edges))))

    clustering = torch.empty(nx.number_of_nodes(graph), dtype=torch.float)
    cuda_kernels.graph_clustering(edges_tensor, clustering)

    return clustering


def feature_extraction(G: nx.graph) -> np.ndarray:
    node_list = sorted(G.nodes())
    node_degree_dict = dict(G.degree())
    clusts = graph_clustering_cuda(G)
    egonets = {n: nx.ego_graph(G, n) for n in node_list}
    degs = [node_degree_dict[n] for n in node_list]

    neighbor_degs = [
        np.mean([node_degree_dict[m] for m in egonets[n].nodes if m != n])
        if node_degree_dict[n] > 0
        else 0
        for n in node_list
    ]

    neighbor_clusts = [
        np.mean([clusts[m] for m in egonets[n].nodes if m != n])
        if node_degree_dict[n] > 0
        else 0
        for n in node_list
    ]

    return np.nan_to_num(np.stack((degs, clusts, neighbor_degs, neighbor_clusts)).T)
