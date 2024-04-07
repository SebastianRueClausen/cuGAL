import unittest
from cugal.feature_extraction import graph_clustering, graph_clustering_cuda
import networkx as nx
import numpy as np


def random_graph(size: int) -> nx.graph:
    return nx.newman_watts_strogatz_graph(size, 10, 0.1)


class TestAdjacency(unittest.TestCase):
    def test_graph_clustering_agree(self):
        graph = random_graph(128)
        correct = graph_clustering(graph)
        cuda = graph_clustering_cuda(graph)
        assert np.allclose(correct, cuda)
