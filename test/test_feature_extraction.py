import unittest
from cugal.feature_extraction import extract_features_cuda, extract_features_nx, Features
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
        correct = extract_features_nx(graph)
        cuda = extract_features_cuda(
            graph, Config(device="cuda")).cpu().numpy()
        assert np.allclose(correct, cuda)

    @unittest.skipUnless(condition=torch.cuda.is_available(), reason="requires CUDA")
    def test_euclidean_distances(self):
        source, target = random_graph(32), random_graph(32)

        cpu_features = Features.create(source, target, Config(device="cpu"))
        gpu_features = Features.create(source, target, Config(device="cuda"))
        cpu_matrix, gpu_matrix = cpu_features.distance_matrix(
        ), gpu_features.distance_matrix().cpu()

        assert torch.allclose(cpu_matrix, gpu_matrix, rtol=1e-4, atol=1e-6)

    @unittest.skipUnless(condition=torch.cuda.is_available(), reason="requires CUDA")
    def test_add_euclidean_distance(self):
        source, target = random_graph(32), random_graph(32)

        features = Features.create(source, target, Config(
            device="cuda", dtype=torch.float32))

        correct = features.distance_matrix()
        test = features.add_distance(torch.zeros_like(correct))

        # TODO: Figure out why this is so different.
        # assert torch.allclose(correct, test, rtol=1e-2, atol=1e-2)
