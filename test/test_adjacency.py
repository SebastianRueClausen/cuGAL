"""Test ajacency stuff."""

import unittest
from cugal.adjacency import Adjacency
from cugal.pred import gradient, sparse_gradient
import torch
import random
import networkx as nx


def random_adjacency_matrix(size: int, device="cpu") -> torch.Tensor:
    return torch.randint(2, size=(size, size), dtype=torch.float32, device=device)


class TestAdjacency(unittest.TestCase):
    def test_conversion_torch(self):
        for _ in range(32):
            size = random.randint(10, 255)
            dense = torch.randint(2, size=(size, size), dtype=torch.float32)
            converted_dense = Adjacency.from_dense(
                dense).as_dense(torch.float32)
            assert torch.all(dense == converted_dense)

    def test_gradient_torch(self):
        size = random.randint(16, 32)
        A, B = random_adjacency_matrix(size), random_adjacency_matrix(size)

        P = torch.randn(size=(size, size), dtype=torch.float32)
        K = torch.randn(size=(size, size), dtype=torch.float32)

        grad = gradient(A, B, P, K, 0)
        sparse_grad = sparse_gradient(Adjacency.from_dense(A), Adjacency.from_dense(
            B), Adjacency.from_dense(A.T), Adjacency.from_dense(B.T), P, K, 0)

        # This requres a bit high error tolerance.
        assert torch.allclose(grad, sparse_grad, rtol=1e-4, atol=1e-5)

    def test_from_graph(self):
        graph = nx.newman_watts_strogatz_graph(16, 7, 0.01)
        dense = torch.from_numpy(nx.to_numpy_array(graph)).to(torch.float32)
        assert torch.allclose(
            dense, Adjacency.from_graph(graph, "cpu").as_dense(torch.float32))

    @unittest.skipUnless(condition=torch.cuda.is_available(), reason="requires CUDA")
    def test_matmul_cuda(self):
        size = 8
        A = random_adjacency_matrix(size, "cuda")
        dense = torch.randn(size=(size, size),
                            dtype=torch.float32, device="cuda")

        truth = A @ dense
        test = Adjacency.from_dense(A).mul(dense)

        assert torch.allclose(truth, test)

    @unittest.skipUnless(condition=torch.cuda.is_available(), reason="requires CUDA")
    def test_gradient_cuda(self):
        size = random.randint(16, 32)
        A, B = random_adjacency_matrix(
            size, "cuda"), random_adjacency_matrix(size, "cuda")

        P = torch.randn(size=(size, size), dtype=torch.float32, device="cuda")
        K = torch.randn(size=(size, size), dtype=torch.float32, device="cuda")

        grad = gradient(A, B, P, K, 0)
        sparse_grad = sparse_gradient(Adjacency.from_dense(A), Adjacency.from_dense(
            B), Adjacency.from_dense(A.T), Adjacency.from_dense(B.T), P, K, 0)

        # This requres a bit high error tolerance.
        assert torch.allclose(grad, sparse_grad, rtol=1e-4, atol=1e-5)
