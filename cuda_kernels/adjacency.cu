#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <torch/torch.h>
#include "common.cuh"

template <typename index_t>
__global__ void kernel(
    const Accessor<index_t, 1> col_indices,
    const Accessor<int, 1> row_pointers,
    const Accessor<float, 2> matrix,
    Accessor<float, 2> out,
    int size,
    float sign
) {
    const auto tid = threadIdx.x;

    const auto row = blockIdx.x;
    const auto col = blockIdx.y;

    const auto start = row_pointers[row];
    const auto end = row == size - 1 ? col_indices.size(0) : row_pointers[row + 1];

    float sum = 0.0;

    for (auto i = start + tid; i < end; i += blockDim.x) {
        sum = fma(sign, matrix[col_indices[i]][col], sum);
    }

    sum = warp_sum_reduce(sum);

    if (tid == 0) {
        out[row][col] = sum;
    }
}

constexpr size_t block_size = 32;

void adjacency_matmul_cuda(
    torch::Tensor col_indices,
    torch::Tensor row_pointers,
    torch::Tensor matrix,
    torch::Tensor out,
    bool negate_lhs
) {
    const dim3 blocks(matrix.size(0), matrix.size(1), 1);
    const auto size = matrix.size(0);

    const auto sign = negate_lhs ? -1.0 : 1.0;

    if (col_indices.scalar_type() == torch::ScalarType::Int) {
        kernel<int><<<blocks, block_size>>>(
            col_indices.packed_accessor32<int, 1>(),
            row_pointers.packed_accessor32<int, 1>(),
            matrix.packed_accessor32<float, 2>(),
            out.packed_accessor32<float, 2>(),
            size,
            sign
        );
    } else if (col_indices.scalar_type() == torch::ScalarType::Short) {
        kernel<short><<<blocks, block_size>>>(
            col_indices.packed_accessor32<short, 1>(),
            row_pointers.packed_accessor32<int, 1>(),
            matrix.packed_accessor32<float, 2>(),
            out.packed_accessor32<float, 2>(),
            size,
            sign
        );
    } else {
        printf("invalid data type\n");
        abort();
    }

    cudaDeviceSynchronize();
}

//
// Create adjacency.
//

#include <thrust/sort.h>
#include <thrust/device_vector.h>

template <typename index_t>
struct Edge {
    index_t row, col;

    __host__ __device__ bool operator <(Edge<index_t> const& rhs) {
        if (row == rhs.row) {
            return col < rhs.col;
        } else {
            return row < rhs.row;
        }
    }
};

template <typename index_t>
inline bool operator<(const Edge<index_t>& lhs, const Edge<index_t>& rhs) {
    return lhs < rhs;
}

// Go through `edges` in parallel. When a thread sees something like
// "(0, 0), (1, 0)", it knows where row 1 starts, which it can write to
// `row_pointers` without race conditions.
template <typename index_t>
__global__ void create_row_pointers(
    const Accessor<Edge<index_t>, 1> edges,
    Accessor<int, 1> row_pointers
) {
    const auto tid = threadIdx.x + blockIdx.x * blockDim.x;
    const auto stride = blockDim.x * gridDim.x;

    const auto size = edges.size(0);

    // Go through all edges and check where if there is a change in the row.
    for (auto edge_index = tid; edge_index < size - 1; edge_index += stride) {
        const auto current = edges[edge_index];
        const auto next = edges[edge_index + 1];

        // If `current` and `next` have different rows, we know that every row between
        // the two and `next.row` starts on `edge_index + 1`.
        for (auto row_index = current.row + 1; row_index <= next.row; row_index++) {
            row_pointers[row_index] = edge_index + 1;
        }
    }
}

// Copy the `col` field of each `Each` in `edges` into `col_indices`.
template<typename index_t>
__global__ void create_col_indices(
    const Accessor<Edge<index_t>, 1> edges,
    Accessor<index_t, 1> col_indices
) {
    const auto tid = threadIdx.x + blockIdx.x * blockDim.x;
    const auto stride = blockDim.x * gridDim.x;

    const auto size = edges.size(0);

    for (auto edge_index = tid; edge_index < size; edge_index += stride) {
        col_indices[edge_index] = edges[edge_index].col;
    }
}

int div_ceil(int x, int y) {
  return (x + y - 1) / y;
}

void create_adjacency_cuda(torch::Tensor edges, torch::Tensor col_indices, torch::Tensor row_pointers) {
    constexpr auto block_size = 64;
    constexpr auto thread_count = 1024;

    const auto edge_count = edges.size(0) / 2;
    const auto block_count = div_ceil(thread_count, block_size);

    if (edges.scalar_type() == torch::ScalarType::Short) {
        const auto edge_ptr = reinterpret_cast<Edge<short>*>(edges.data_ptr());

        const auto stride = edges.stride(0);
        const auto edges_accessor = Accessor<Edge<short>, 1>(edge_ptr, &edge_count, &stride);

        create_row_pointers<short><<<block_count, block_size>>>(
            edges_accessor, row_pointers.packed_accessor32<int, 1>()
        );
        create_col_indices<short><<<block_count, block_size>>>(
            edges_accessor, col_indices.packed_accessor32<short, 1>()
        );
    } else if (edges.scalar_type() == torch::ScalarType::Int) {
        const auto edge_ptr = reinterpret_cast<Edge<int>*>(edges.data_ptr());

        const auto stride = edges.stride(0);
        const auto edges_accessor = Accessor<Edge<int>, 1>(edge_ptr, &edge_count, &stride);

        create_row_pointers<int><<<block_count, block_size>>>(
            edges_accessor, row_pointers.packed_accessor32<int, 1>()
        );
        create_col_indices<int><<<block_count, block_size>>>(
            edges_accessor, col_indices.packed_accessor32<int, 1>()
        );
    }

    cudaDeviceSynchronize();
}
