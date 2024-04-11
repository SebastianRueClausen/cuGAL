#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <torch/torch.h>
#include <cub/cub.cuh>
#include <iostream>
#include "common.cuh"
#include <cusparse.h>

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
    const auto end = row_pointers[row + 1];

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
}


#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
    }                                                                          \
}

auto cusparse_data_type(torch::ScalarType scalar_type) {
    switch (scalar_type) {
        case torch::ScalarType::Half:
            return CUDA_R_16F;
        case torch::ScalarType::Double:
            return CUDA_R_64F;
        default:
            return CUDA_R_32F;
    }
}

void adjacency_matmul_cusparse(
    torch::Tensor col_indices,
    torch::Tensor row_pointers,
    torch::Tensor matrix,
    torch::Tensor out,
    bool negate_lhs
) {
    const auto size = matrix.size(0);

    // The cusparse needs a value vectors, so we just create one full of ones.
    const auto fill_value = negate_lhs ? -1.0f : 1.0f;
    const auto options = torch::TensorOptions()
        .dtype(matrix.dtype())
        .device(matrix.device());
    const auto values = torch::full(col_indices.sizes(), fill_value, options);

    const auto scalar_type = cusparse_data_type(values.scalar_type());
    const auto index_type = CUSPARSE_INDEX_32I;
    const auto index_base = CUSPARSE_INDEX_BASE_ZERO;

    // FIXME: This is kind of bad. We just assume that matrix is transposed if it isn't contigous.
    const auto rhs_transpose = !matrix.is_contiguous();

    // Create handle for cusparse.
    cusparseHandle_t handle = nullptr;
    cusparseCreate(&handle);

    // Create cusparse sparse matrix.
    cusparseSpMatDescr_t sparse_desc;
    CHECK_CUSPARSE(cusparseCreateCsr(
        &sparse_desc, size, size, col_indices.size(0), row_pointers.data_ptr(), col_indices.data_ptr(), values.data_ptr(),
        index_type, index_type, index_base, scalar_type
    ));

    const auto order = CUSPARSE_ORDER_ROW;

    // Create cusparse dense matrices.
    cusparseDnMatDescr_t dense_desc, out_desc;
    CHECK_CUSPARSE(cusparseCreateDnMat(
        &dense_desc, size, size, size, matrix.data_ptr(), scalar_type, CUSPARSE_ORDER_ROW
    ));
    CHECK_CUSPARSE(cusparseCreateDnMat(
        &out_desc, size, size, size, out.data_ptr(), scalar_type,
        // TODO: Figure out why this is this required?.
        rhs_transpose ? CUSPARSE_ORDER_COL : CUSPARSE_ORDER_ROW
    ));

    // Values multiplied with the sparse matrix.
    const auto alpha = 1.0f;
    // Value multiplied with out matrix before the result being added.
    const auto beta = 0.0f;

    const auto lhs_op = CUSPARSE_OPERATION_NON_TRANSPOSE;
    const auto rhs_op = matrix.is_contiguous()
        ? CUSPARSE_OPERATION_NON_TRANSPOSE : CUSPARSE_OPERATION_TRANSPOSE;
    const auto algo = CUSPARSE_SPMM_ALG_DEFAULT;

    // Determine size of buffer.
    size_t buffer_size;
    CHECK_CUSPARSE(cusparseSpMM_bufferSize(
        handle, lhs_op, rhs_op, &alpha, sparse_desc, dense_desc, &beta, out_desc, scalar_type, algo, &buffer_size
    ));

    void* buffer = nullptr;
    cudaMalloc(&buffer, buffer_size);

    // Do multiplication.
    CHECK_CUSPARSE(cusparseSpMM(
        handle, lhs_op, rhs_op, &alpha, sparse_desc, dense_desc, &beta, out_desc, scalar_type, algo, buffer
    ));

    cudaDeviceSynchronize();

    cusparseDestroySpMat(sparse_desc);
    cusparseDestroyDnMat(dense_desc);
    cusparseDestroyDnMat(out_desc);
    cusparseDestroy(handle);

    cudaFree(buffer);
}

//
// Create adjacency.
//

template <typename index_t>
struct Edge {
    index_t col, row;

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
    const auto row_count = row_pointers.size(0);

    if (tid == 0) {
        row_pointers[0] = 0;
        row_pointers[row_count - 1] = size;
    }

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

template <typename key_t>
void sort_edges(key_t* edges, key_t* sorted, long edge_count) {
    // This is only to obtain the required temporary storage size.
    size_t tmp_storage_size = 0;
    cub::DeviceRadixSort::SortKeys<key_t>(nullptr, tmp_storage_size, edges, sorted, edge_count);

    // Allocate temporary storage.
    void* tmp_storage = nullptr;
    cudaMalloc(&tmp_storage, tmp_storage_size);

    // Sort edges.
    cub::DeviceRadixSort::SortKeys<key_t>(tmp_storage, tmp_storage_size, edges, sorted, edge_count);
    cudaDeviceSynchronize();

    // Deallocate temporary storage.
    cudaFree(tmp_storage);
}

void create_adjacency_cuda(torch::Tensor edges, torch::Tensor col_indices, torch::Tensor row_pointers) {
    constexpr auto block_size = 64;
    constexpr auto thread_count = 1024;

    const auto edge_count = edges.size(0) / 2;
    constexpr auto block_count = div_ceil(thread_count, block_size);

    auto sorted_edges = torch::empty_like(edges);

    if (edges.scalar_type() == torch::ScalarType::Short) {
        const auto sorted_edge_ptr = reinterpret_cast<Edge<short>*>(sorted_edges.data_ptr());

        sort_edges<uint32_t>((uint32_t*)edges.data_ptr(), (uint32_t*)sorted_edge_ptr, edge_count);

        const auto stride = edges.stride(0);
        const auto edges_accessor = Accessor<Edge<short>, 1>(sorted_edge_ptr, &edge_count, &stride);

        create_row_pointers<short><<<block_count, block_size>>>(
            edges_accessor, row_pointers.packed_accessor32<int, 1>()
        );
        create_col_indices<short><<<block_count, block_size>>>(
            edges_accessor, col_indices.packed_accessor32<short, 1>()
        );
    } else if (edges.scalar_type() == torch::ScalarType::Int) {
        const auto sorted_edge_ptr = reinterpret_cast<Edge<int>*>(sorted_edges.data_ptr());

        sort_edges<uint64_t>((uint64_t*)edges.data_ptr(), (uint64_t*)sorted_edge_ptr, edge_count);

        const auto stride = edges.stride(0);
        const auto edges_accessor = Accessor<Edge<int>, 1>(sorted_edge_ptr, &edge_count, &stride);

        create_row_pointers<int><<<block_count, block_size>>>(
            edges_accessor, row_pointers.packed_accessor32<int, 1>()
        );
        create_col_indices<int><<<block_count, block_size>>>(
            edges_accessor, col_indices.packed_accessor32<int, 1>()
        );
    }

    cudaDeviceSynchronize();
}
