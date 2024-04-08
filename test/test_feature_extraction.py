import unittest
from cugal.feature_extraction import extract_features_cuda, extract_features, feature_distance_matrix
from cugal.config import Config
import networkx as nx
import numpy as np
import torch


def random_graph(size: int) -> nx.graph:
    return nx.newman_watts_strogatz_graph(size, 10, 0.1)


class TestFeatureExtraction(unittest.TestCase):
    @unittest.skipUnless(condition=torch.cuda.is_available(), reason="requires CUDA")
    def test_graph_features(self):
        graph = random_graph(32)
        correct = extract_features(graph)
        cuda = extract_features_cuda(
            graph, Config(device="cuda")).cpu().numpy()
        assert np.allclose(correct, cuda)

    @unittest.skipUnless(condition=torch.cuda.is_available(), reason="requires CUDA")
    def test_euclidian_distances(self):
        source, target = random_graph(32), random_graph(32)
        cpu_config = Config(device="cpu")
        cuda_config = Config(device="cuda")
        cpu_matrix = feature_distance_matrix(source, target, cpu_config)
        cuda_matrix = feature_distance_matrix(
            source, target, cuda_config).cpu()
        assert torch.allclose(cpu_matrix, cuda_matrix, rtol=1e-4, atol=1e-6)
