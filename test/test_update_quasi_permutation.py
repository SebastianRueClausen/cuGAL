import unittest
from cugal.config import SinkhornMethod
from cugal.pred import update_quasi_permutation
import torch

try:
    import cuda_kernels
    has_cuda = True
except ImportError:
    has_cuda = False


def assert_agree(sinkhorn_method: SinkhornMethod):
    n = 128
    P = torch.randn((n, n), dtype=torch.float32, device='cuda')
    K = torch.randn((n, n), dtype=torch.float32, device='cuda')
    u, v = torch.randn(n, dtype=torch.float32, device='cuda'), torch.randn(
        n, dtype=torch.float32, device='cuda')
    K = torch.randn((n, n), dtype=torch.float32, device='cuda')
    alpha = 0.2
    cuda_P, torch_P = P.clone(), P.clone()
    cuda_kernels.update_quasi_permutation_log(
        cuda_P, K, u, v, alpha, 0.0)
    update_quasi_permutation(torch_P, K, u, v, alpha, sinkhorn_method)
    assert torch.allclose(cuda_P, torch_P, rtol=0.0001, atol=1e-6)


class TestUpdateQuasiPermutation(unittest.TestCase):
    @unittest.skipUnless(condition=has_cuda, reason="requires cuda_kernels installed")
    def test_agree_log(self):
        assert_agree(SinkhornMethod.LOG)
