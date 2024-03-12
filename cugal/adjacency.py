"""Representation of an adjacency matrix."""

from dataclasses import dataclass
from itertools import product
import numpy as np
import torch
import networkx as nx
from itertools import groupby

try:
    import cuda_kernels
    has_cuda = True
except ImportError:
    has_cuda = False


def determine_index_type(size: int) -> torch.dtype:
    return torch.int16 if size < np.iinfo(np.int16).max else torch.int32


@dataclass
class Adjacency:
    col_indices: torch.Tensor
    """Indices into rows where there are ones."""

    row_pointers: torch.Tensor
    """Indices into col_indices each row begins."""

    @classmethod
    def from_dense(cls, dense: torch.Tensor):
        """Create from from dense adjacency matrix."""

        size = len(dense)
        dtype = determine_index_type(size)

        col_indices = torch.tensor(
            [], dtype=dtype, device=dense.device)
        row_pointers = torch.empty(
            size=(size,), dtype=torch.int32, device=dense.device)

        for row_index, row in enumerate(dense):
            row_pointers[row_index] = len(col_indices)
            col_indices = torch.cat((col_indices, row.nonzero().to(dtype)))

        return cls(col_indices.squeeze(1), row_pointers)

    @classmethod
    def from_graph(cls, graph: nx.graph, device: torch.device):
        """Create from networkx graph."""

        node_count = graph.number_of_nodes()
        dtype = determine_index_type(node_count)

        edges = list(nx.edges(graph))
        if not graph.is_directed():
            edges += [(y, x) for x, y in edges]

        if has_cuda and "cuda" in str(device):
            edges = torch.tensor(sum(edges, ()), dtype=dtype, device=device)
            col_indices = torch.empty(
                size=(len(edges),), dtype=dtype, device=device)
            row_pointers = torch.empty(
                size=(node_count,), dtype=torch.int, device=device)

            cuda_kernels.create_adjacency(edges, col_indices, row_pointers)

            return cls(col_indices, row_pointers)

        col_indices, row_pointers = \
            torch.empty(size=(len(edges),)), torch.empty(size=(node_count,))

        col_index_count, row_index = 0, 0
        for node_index, node_edges in groupby(sorted(edges), key=lambda edge: edge[0]):
            while row_index <= node_index:
                row_pointers[row_index] = col_index_count
                row_index += 1
            for _, to in node_edges:
                col_indices[col_index_count] = to
                col_index_count += 1

        while row_index < len(row_pointers):
            row_pointers[row_index] = col_index_count
            row_index += 1

        col_indices = col_indices.to(dtype=dtype, device=device)
        row_pointers = row_pointers.to(dtype=torch.int32, device=device)
        return cls(col_indices, row_pointers)

    def as_dense(self, dtype: torch.dtype) -> torch.Tensor:
        """Convert back to dense representation."""

        col_indices = torch.clone(self.col_indices).to(torch.int64)
        dense = torch.zeros((self.size(), self.size()), dtype=dtype)

        for row_index, begin in enumerate(self.row_pointers):
            end = len(col_indices) if row_index == self.size() - \
                1 else self.row_pointers[row_index+1]
            dense[row_index, :].scatter_(
                0, col_indices[begin:end], 1)

        return dense

    def byte_size(self) -> int:
        element_size = 2 if self.col_indices.dtype == torch.int16 else torch.int32
        return element_size * len(self.col_indices) + 4 * len(self.row_pointers)

    def size(self) -> int:
        return len(self.row_pointers)

    def mul(self, matrix: torch.Tensor, negate_lhs: bool = False) -> torch.Tensor:
        """Calculate self @ matrix."""

        assert matrix.shape[0] == self.size(
        ) and matrix.shape[1] == self.size(), "matrix must match size"

        use_cuda = \
            has_cuda and "cuda" in str(
                matrix.device) and matrix.dtype == torch.float32

        out = torch.empty_like(matrix)

        if use_cuda:
            cuda_kernels.adjacency_matmul(
                self.col_indices, self.row_pointers, matrix, out, negate_lhs,
            )
        else:
            for row_index, col_index in product(range(self.size()), repeat=2):
                start = self.row_pointers[row_index]
                end = len(self.col_indices) if row_index == self.size() - \
                    1 else self.row_pointers[row_index+1]
                indices = self.col_indices[start:end].to(torch.int32)
                out[row_index, col_index] = \
                    torch.sum(
                        matrix[indices, col_index] * (-1 if negate_lhs else 1))

        return out
