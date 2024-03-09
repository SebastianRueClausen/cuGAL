"""Representation of an adjacency matrix."""

from dataclasses import dataclass
import torch

try:
    import cuda_kernels
    has_cuda = True
except ImportError:
    has_cuda = False


@dataclass
class Adjacency:
    col_indices: torch.Tensor
    """Indices into rows where there are ones."""

    row_pointers: torch.Tensor
    """Indices into col_indices each row begins."""

    def __init__(self, dense: torch.Tensor):
        """Create from from dense adjacency matrix."""

        size = len(dense)
        self.col_indices = torch.tensor(
            [], dtype=torch.int32, device=dense.device)
        self.row_pointers = torch.empty(
            size=(size,), dtype=torch.int32, device=dense.device)

        for row_index, row in enumerate(dense):
            col_indices = row.nonzero().to(torch.int32)
            self.row_pointers[row_index] = len(self.col_indices)
            self.col_indices = torch.cat((self.col_indices, col_indices))

        self.col_indices = self.col_indices.squeeze(1)

    def as_dense(self, dtype: torch.dtype) -> torch.Tensor:
        """Convert back to dense representation."""

        col_indices = torch.clone(self.col_indices).to(torch.int64)

        size = len(self.row_pointers)
        dense = torch.zeros((size, size), dtype=dtype)

        for row_index, begin in enumerate(self.row_pointers):
            end = len(col_indices) if row_index == size - \
                1 else self.row_pointers[row_index+1]
            dense[row_index, :].scatter_(
                0, col_indices[begin:end], 1)

        return dense

    def mul(self, matrix: torch.Tensor, negate_lhs: bool = False) -> torch.Tensor:
        """Calculate self @ matrix."""

        use_cuda = \
            has_cuda and "cuda" in str(
                matrix.device) and matrix.dtype == torch.float32

        if use_cuda:
            out = torch.empty_like(matrix)
            cuda_kernels.adjacency_matmul(
                self.col_indices,
                self.row_pointers,
                matrix,
                out,
                negate_lhs
            )
        else:
            lhs = self.as_dense(matrix.dtype)
            if negate_lhs:
                lhs = -lhs
            out = lhs @ matrix

        return out
