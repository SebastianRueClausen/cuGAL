"""Test implementations of Sinkhorn-Knopp."""

import unittest
import cugal.sinkhorn
from cugal import SinkhornMethod, Config
import torch


def sinkhorn(C: torch.Tensor, config: Config) -> torch.Tensor:
    scale = cugal.sinkhorn.scale_kernel_matrix_log if config.sinkhorn_method == SinkhornMethod.LOG else cugal.sinkhorn.scale_kernel_matrix
    return scale(*cugal.sinkhorn.sinkhorn(C, config))


def random_matrix(size: int) -> torch.Tensor:
    return torch.rand((size, size)) * 10.0


def assert_is_doubly_stochastic(matrix: torch.Tensor, rtol: float = 1e-5, atol: float = 1e-8):
    ones = torch.ones((matrix.shape[0], ),
                      device=matrix.device, dtype=matrix.dtype)
    assert torch.allclose(ones, torch.sum(matrix, dim=0), rtol=rtol, atol=atol)
    assert torch.allclose(ones, torch.sum(matrix, dim=1), rtol=rtol, atol=atol)


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

    def test_mix_cpu_float32_agree(self):
        test_config = Config(sinkhorn_method=SinkhornMethod.MIX,
                             dtype=torch.float32)
        truth_config = Config()
        matrix = random_matrix(128)
        truth = sinkhorn(truth_config.convert_tensor(matrix), truth_config)
        test = sinkhorn(test_config.convert_tensor(matrix), test_config)
        assert torch.allclose(
            truth, truth_config.convert_tensor(test), rtol=1e-4, atol=1e-6)

    def test_mix_cpu_float32_doubly_stochastic(self):
        test_config = Config(sinkhorn_method=SinkhornMethod.MIX,
                             dtype=torch.float32)
        matrix = random_matrix(128)
        test = sinkhorn(test_config.convert_tensor(matrix), test_config)
        assert_is_doubly_stochastic(test, rtol=1e-3, atol=1e-6)

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
                             dtype=torch.float32, device="cuda")
        matrix = random_matrix(128)
        test = sinkhorn(test_config.convert_tensor(matrix), test_config)
        assert_is_doubly_stochastic(test)
