import networkx as nx
import numpy as np
import torch
from cugal.adjacency import Adjacency
from cugal.config import Config
from dataclasses import dataclass
import math

try:
    import cuda_kernels
    has_cuda = True
except ImportError as e:
    print("IMPORT ERROR:\n", e)
    has_cuda = False


def extract_features_cuda(graph: nx.Graph | Adjacency, config: Config) -> torch.Tensor:
    assert 'cuda' in config.device
    adjacency = graph if type(
        graph) == Adjacency else Adjacency.from_graph(graph, config.device)
    clustering = torch.zeros(nx.number_of_nodes(
        graph), dtype=torch.float, device=config.device)
    degrees = torch.zeros(nx.number_of_nodes(
        graph), dtype=torch.float, device=config.device)
    neighbor_clustering = torch.zeros(nx.number_of_nodes(
        graph), dtype=torch.float, device=config.device)
    neighbor_degrees = torch.zeros(nx.number_of_nodes(
        graph), dtype=torch.float, device=config.device)
    cuda_kernels.graph_features(
        adjacency.col_indices, adjacency.row_pointers, clustering, degrees)
    torch.cuda.synchronize()
    cuda_kernels.average_neighbor_features(
        adjacency.col_indices, adjacency.row_pointers, clustering, neighbor_clustering)
    cuda_kernels.average_neighbor_features(
        adjacency.col_indices, adjacency.row_pointers, degrees, neighbor_degrees)
    torch.cuda.synchronize()
    return torch.stack((degrees, clustering, neighbor_degrees, neighbor_clustering)).T


def extract_features_nx(G: nx.Graph) -> np.ndarray:
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


@dataclass
class Features:
    """Features of source and target graph."""

    source: torch.Tensor
    target: torch.Tensor

    @classmethod
    def create(cls, source: nx.Graph | Adjacency, target: nx.Graph | Adjacency, config: Config):
        """Calculate features of the source and target graph."""

        use_cuda = has_cuda and 'cuda' in config.device
        print(has_cuda, config.device)

        if use_cuda:
            source_features = extract_features_cuda(
                source, config).to(config.dtype)
            target_features = extract_features_cuda(
                target, config).to(config.dtype)
        else:
            source_features = torch.tensor(extract_features_nx(
                source), device=config.device, dtype=config.dtype)
            target_features = torch.tensor(extract_features_nx(
                target), device=config.device, dtype=config.dtype)

        source_features *= math.sqrt(config.mu)
        target_features *= math.sqrt(config.mu)

        return cls(source_features, target_features)

    def add_distance(self, out: torch.Tensor) -> torch.Tensor:
        """Calculate `out + self.distance_matrix() * config.mu` efficiently."""

        if has_cuda and 'cuda' in str(out.device):
            cuda_kernels.add_distance(self.source, self.target, out)
        else:
            out += self.distance_matrix()

        return out

    def distance_matrix(self) -> torch.Tensor:
        """Calculate euclidean distance matrix."""
        return torch.cdist(self.source, self.target)
