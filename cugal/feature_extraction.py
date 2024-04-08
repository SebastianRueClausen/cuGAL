import networkx as nx
import numpy as np
import cuda_kernels
import torch
from cugal.adjacency import Adjacency


def graph_clustering(graph: nx.graph) -> np.ndarray:
    clustering = nx.clustering(graph)
    return np.array([clustering[i] for i in graph.nodes()])


def graph_degree(graph: nx.graph) -> np.ndarray:
    degrees = dict(nx.degree(graph))
    return np.array([degrees[i] for i in graph.nodes()])


def extract_features_cuda(graph: nx.graph) -> torch.Tensor:
    adjacency = Adjacency.from_graph(graph, "cuda")
    clustering = torch.zeros(nx.number_of_nodes(
        graph), dtype=torch.float, device="cuda")
    degrees = torch.zeros(nx.number_of_nodes(
        graph), dtype=torch.float, device="cuda")
    neighbor_clustering = torch.zeros(nx.number_of_nodes(
        graph), dtype=torch.float, device="cuda")
    neighbor_degrees = torch.zeros(nx.number_of_nodes(
        graph), dtype=torch.float, device="cuda")
    cuda_kernels.graph_features(
        adjacency.col_indices, adjacency.row_pointers, clustering, degrees)
    cuda_kernels.average_neighbor_features(
        adjacency.col_indices, adjacency.row_pointers, clustering, neighbor_clustering)
    cuda_kernels.average_neighbor_features(
        adjacency.col_indices, adjacency.row_pointers, degrees, neighbor_degrees)
    return torch.stack((degrees, clustering, neighbor_degrees, neighbor_clustering)).T


def extract_features(G: nx.graph) -> np.ndarray:
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
