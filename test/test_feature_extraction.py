import unittest
from cugal.feature_extraction import graph_clustering, graph_features_cuda, graph_degree
import networkx as nx
import numpy as np


def random_graph(size: int) -> nx.graph:
    return nx.newman_watts_strogatz_graph(size, 10, 0.1)


class TestFeatureExtraction(unittest.TestCase):
    def test_graph_features(self):
        graph = random_graph(32)

        correct_clustering = graph_clustering(graph)
        correct_degrees = graph_degree(graph)

        clustering, degrees = graph_features_cuda(graph)
        clustering, degrees = clustering.cpu().numpy(), degrees.cpu().numpy()

        assert np.allclose(correct_clustering, clustering)
        assert np.allclose(correct_degrees, degrees)
