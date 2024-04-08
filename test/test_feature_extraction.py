import unittest
from cugal.feature_extraction import extract_features_cuda, extract_features
import networkx as nx
import numpy as np


def random_graph(size: int) -> nx.graph:
    return nx.newman_watts_strogatz_graph(size, 10, 0.1)


class TestFeatureExtraction(unittest.TestCase):
    def test_graph_features(self):
        graph = random_graph(32)
        correct = extract_features(graph)
        cuda = extract_features_cuda(graph).cpu().numpy()
        assert np.allclose(correct, cuda)
