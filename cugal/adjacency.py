from dataclasses import dataclass
import torch


@dataclass
class Adjacency:
    col_indices: torch.Tensor
    row_pointers: torch.Tensor

    def __init__(self, dense: torch.Tensor):
        """Create from from dense adjacency matrix."""

        size = len(dense)
        self.col_indices = torch.tensor([], dtype=torch.int32)
        self.row_pointers = torch.empty(size=(size,), dtype=torch.int32)

        for row_index, row in enumerate(dense):
            col_indices = row.nonzero()
            self.row_pointers[row_index] = len(self.col_indices)
            self.col_indices = torch.cat((self.col_indices, col_indices))

    def as_dense(self, dtype: torch.dtype) -> torch.Tensor:
        """Convert back to dense representation."""

        size = len(self.row_pointers)
        dense = torch.zeros((size, size), dtype=dtype)

        for row_index, begin in enumerate(self.row_pointers):
            end = len(self.col_indices) if row_index == size - \
                1 else self.row_pointers[row_index+1]
            dense[row_index, :].scatter_(
                0, self.col_indices[begin:end].squeeze(1), 1)

        return dense

    def mul(self, matrix: torch.Tensor, negate_lhs: bool = False) -> torch.Tensor:
        """Calculate self @ matrix."""

        lhs = self.as_dense(matrix.dtype)
        if negate_lhs:
            lhs = -lhs
        return lhs @ matrix
