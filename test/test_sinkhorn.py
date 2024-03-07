"""Test implementations of Sinkhorn-Knopp."""

import unittest
from cugal.sinkhorn import sinkhorn
from cugal import SinkhornMethod, Config
import torch


def random_matrix(size: int) -> torch.Tensor:
    return torch.rand((size, size)) * 10.0


def assert_is_doubly_stochastic(matrix: torch.Tensor):
    assert torch.allclose(torch.ones_like(matrix), torch.sum(matrix, dim=0))
    assert torch.allclose(torch.ones_like(matrix), torch.sum(matrix, dim=1))


class TestSinkhorn(unittest.TestCase):
    def test_log_cpu_float32_agree(self):
        test_config = Config(sinkhorn_method=SinkhornMethod.LOG,
                        dtype=torch.float32)
        truth_config = Config()
        matrix = random_matrix(128)
        truth = sinkhorn(truth_config.convert_tensor(matrix), truth_config)
        test = sinkhorn(test_config.convert_tensor(matrix), test_config)
        assert torch.allclose(truth, truth_config.convert_tensor(test))

    def test_log_cpu_float32_doubly_stochastic(self):
        test_config = Config(sinkhorn_method=SinkhornMethod.LOG,
                        dtype=torch.float32)
        matrix = random_matrix(128)
        test = sinkhorn(test_config.convert_tensor(matrix), test_config)
        assert_is_doubly_stochastic(test)

    @unittest.skipUnless(condition=torch.cuda.is_available(), reason="requires CUDA")
    def test_log_cuda_float64_doubly_stochastic(self):
        test_config = Config(sinkhorn_method=SinkhornMethod.LOG,
                        dtype=torch.float64)
        matrix = random_matrix(128)
        test = sinkhorn(test_config.convert_tensor(matrix), test_config)
        assert_is_doubly_stochastic(test)

    @unittest.skipUnless(condition=torch.cuda.is_available(), reason="requires CUDA")
    def test_log_cuda_float32_agree(self):
        test_config = Config(sinkhorn_method=SinkhornMethod.LOG,
                        dtype=torch.float32, device="cuda")
        truth_config = Config()
        matrix = random_matrix(128)
        truth = sinkhorn(truth_config.convert_tensor(matrix), truth_config)
        test = sinkhorn(test_config.convert_tensor(matrix), test_config)
        assert torch.allclose(truth, truth_config.convert_tensor(test))

    @unittest.skipUnless(condition=torch.cuda.is_available(), reason="requires CUDA")
    def test_log_cuda_float32_doubly_stochastic(self):
        test_config = Config(sinkhorn_method=SinkhornMethod.LOG,
                        dtype=torch.float32, device="cuda")
        matrix = random_matrix(128)
        test = sinkhorn(test_config.convert_tensor(matrix), test_config)
        assert_is_doubly_stochastic(test)

    @unittest.skipUnless(condition=torch.cuda.is_available(), reason="requires CUDA")
    def test_log_cuda_float32_doubly_stochastic(self):
        test_config = Config(sinkhorn_method=SinkhornMethod.LOG,
                        dtype=torch.float32, device="cuda:0")
        matrix = random_matrix(128)
        test = sinkhorn(test_config.convert_tensor(matrix), test_config)
        assert_is_doubly_stochastic(test)