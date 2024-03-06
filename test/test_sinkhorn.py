import unittest
from cugal.sinkhorn import sinkhorn
from cugal.config import SinkhornMethod, Config
import torch


def random_matrix(size: int) -> torch.Tensor:
    return torch.rand((size, size)) * 10.0


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
        assert torch.allclose(torch.ones_like(test), torch.sum(test, dim=0))
        assert torch.allclose(torch.ones_like(test), torch.sum(test, dim=1))

    @unittest.skipUnless(condition=torch.cuda.is_available(), reason="requires CUDA")
    def test_log_cuda_float32_agree(self):
        test_config = Config(sinkhorn_method=SinkhornMethod.LOG,
                        dtype=torch.float32, device="cuda:0")
        truth_config = Config()
        matrix = random_matrix(128)
        truth = sinkhorn(truth_config.convert_tensor(matrix), truth_config)
        test = sinkhorn(test_config.convert_tensor(matrix), test_config)
        assert torch.allclose(truth, truth_config.convert_tensor(test))

    @unittest.skipUnless(condition=torch.cuda.is_available(), reason="requires CUDA")
    def test_log_cuda_float32_doubly_stochastic(self):
        test_config = Config(sinkhorn_method=SinkhornMethod.LOG,
                        dtype=torch.float32, device="cuda:0")
        matrix = random_matrix(128)
        test = sinkhorn(test_config.convert_tensor(matrix), test_config)
        assert torch.allclose(torch.ones_like(test), torch.sum(test, dim=0))
        assert torch.allclose(torch.ones_like(test), torch.sum(test, dim=1))
