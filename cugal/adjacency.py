"""Representation of an adjacency matrix."""

from dataclasses import dataclass
from itertools import product
import numpy as np
import torch
import networkx as nx
from itertools import groupby
import warnings

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

    @classmethod
    def from_dense(cls, dense: torch.Tensor):
        """Create from from dense adjacency matrix."""

        size = len(dense)
        dtype = torch.int32

        col_indices = torch.tensor(
            [], dtype=dtype, device=dense.device)
        row_pointers = torch.empty(
            size=(size + 1,), dtype=torch.int32, device=dense.device)

        for row_index, row in enumerate(dense):
            row_pointers[row_index] = len(col_indices)
            col_indices = torch.cat((col_indices, row.nonzero().to(dtype)))

        row_pointers[-1] = len(col_indices)

        return cls(col_indices.squeeze(1), row_pointers)

    @classmethod
    def from_graph(cls, graph: nx.Graph, device: torch.device):
        """Create from networkx graph."""

        node_count = graph.number_of_nodes()
        dtype = torch.int32

        edges = list(nx.edges(graph))
        if not graph.is_directed():
            edges += [(y, x) for x, y in edges]

        edge_count = len(edges)

        if has_cuda and "cuda" in str(device):
            flat = [value for edge in edges for value in edge]
            edges = torch.tensor(flat, dtype=dtype, device=device)
            col_indices = torch.empty(
                size=(edge_count,), dtype=dtype, device=device)
            row_pointers = torch.empty(
                size=(node_count + 1,), dtype=torch.int, device=device)

            cuda_kernels.create_adjacency(edges, col_indices, row_pointers)

            return cls(col_indices, row_pointers)

        col_indices, row_pointers = \
            torch.empty(size=(edge_count,)), torch.empty(
                size=(node_count + 1,))

        row_pointers[0], row_pointers[-1] = 0, edge_count

        col_index_count, row_index = 0, 0
        for node_index, node_edges in groupby(sorted(edges), key=lambda edge: edge[0]):
            while row_index <= node_index:
                row_pointers[row_index] = col_index_count
                row_index += 1
            for _, to in node_edges:
                col_indices[col_index_count] = to
                col_index_count += 1

        while row_index < node_count:
            row_pointers[row_index] = col_index_count
            row_index += 1

        col_indices = col_indices.to(dtype=dtype, device=device)
        row_pointers = row_pointers.to(dtype=torch.int32, device=device)
        return cls(col_indices, row_pointers)

    def as_dense(self, dtype: torch.dtype) -> torch.Tensor:
        """Convert back to dense representation."""

        col_indices = torch.clone(self.col_indices).to(torch.int64)
        dense = torch.zeros((self.number_of_nodes(), self.number_of_nodes()),
                            dtype=dtype, device=col_indices.device)

        for row_index, start in enumerate(self.row_pointers[:-1]):
            end = self.row_pointers[row_index+1]
            dense[row_index, :].scatter_(0, col_indices[start:end], 1)

        return dense

    def byte_size(self) -> int:
        return 4 * len(self.col_indices) + 4 * len(self.row_pointers)

    def number_of_nodes(self) -> int:
        return len(self.row_pointers) - 1

    def validate(self):
        col_indices, row_pointers = self.col_indices.cpu(), self.row_pointers.cpu().numpy()

        assert np.all(np.diff(row_pointers) >= 0), "row_pointers isn't sorted"
        assert row_pointers[0] == 0, "row_pointers doesn't start with 0"
        assert row_pointers[-1] == len(
            col_indices), "row_pointers doesn't end correctly"

        for row_index, start in enumerate(row_pointers[:-1]):
            end = row_pointers[row_index + 1]
            cols = col_indices[start:end].numpy()
            assert np.all(np.diff(cols) >= 0), "col_indices isn't sorted"
            assert np.all(cols <= self.number_of_nodes()
                          ), "invalid entries in col_indices"

    def mul(self, matrix: torch.Tensor, negate_lhs: bool = False) -> torch.Tensor:
        """Calculate self @ matrix."""

        assert matrix.shape[0] == self.number_of_nodes(
        ) and matrix.shape[1] == self.number_of_nodes(), "matrix must match size"

        use_cuda = \
            has_cuda and "cuda" in str(
                matrix.device) and matrix.dtype in [torch.float32, torch.float64]

        out = torch.empty_like(matrix)

        if use_cuda:
            cuda_kernels.adjacency_matmul(
                self.col_indices, self.row_pointers, matrix, out, negate_lhs,
            )
        else:
            # values = torch.full_like(self.row_pointers[:-1], fill_value=-1 if negate_lhs else 1.0)
            # out = torch.sparse_csr_tensor(self.row_pointers, self.col_indices, values) @ matrix
            warnings.warn(
                "using sparse adjacency matrices on a device other than cuda is very slow")
            for row_index, col_index in product(range(self.number_of_nodes()), repeat=2):
                start, end = self.row_pointers[row_index], self.row_pointers[row_index+1]
                indices = self.col_indices[start:end].to(torch.int32)
                out[row_index, col_index] = \
                    torch.sum(
                        matrix[indices, col_index] * (-1 if negate_lhs else 1))

        return out
