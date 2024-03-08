"""Test bit matrix utility."""

import unittest
from cugal.pred import to_bit_matrix, from_bit_matrix, bit_matrix_gradient, gradient
import torch
import random

class TestBitMatrix(unittest.TestCase):
    def test_conversion_cpu(self):
        size = random.randint(10, 255)
        matrix = torch.randint(2, size=(size, size), dtype=torch.float32)
        convertex_matrix = from_bit_matrix(to_bit_matrix(matrix), size, torch.float32)
        assert torch.allclose(matrix, convertex_matrix)

    def test_gradient_cpu(self):
        size = random.randint(10, 255)
        A = torch.randint(2, size=(size, size), dtype=torch.float32)
        B = torch.randint(2, size=(size, size), dtype=torch.float32)
        P = torch.randn(size=(size, size), dtype=torch.float32)
        K = torch.randn(size=(size, size), dtype=torch.float32)

        A_bit, B_bit = to_bit_matrix(A), to_bit_matrix(B)
        A_transpose_bit, B_transpose_bit = to_bit_matrix(A.T), to_bit_matrix(B.T)

        grad = gradient(A, B, P, K, 0)
        bit_grad = bit_matrix_gradient(
            A_bit, A_transpose_bit, B_bit, B_transpose_bit, P, K, 0,
        )

        assert torch.allclose(grad, bit_grad)
