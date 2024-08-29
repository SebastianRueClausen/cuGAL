#include "common.cuh"
#include <c10/cuda/CUDAGuard.h>
#include <cub/cub.cuh>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <torch/torch.h>

#define CHECK_CUSPARSE(func)                                                   \
    {                                                                          \
        cusparseStatus_t status = (func);                                      \
        if (status != CUSPARSE_STATUS_SUCCESS) {                               \
            printf(                                                            \
                "cusparse failed at line %d with error: %s (%d)\n", __LINE__,  \
                cusparseGetErrorString(status), status                         \
            );                                                                 \
        }                                                                      \
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

auto scalar_type_size(torch::ScalarType scalar_type) {
    switch (scalar_type) {
    case torch::ScalarType::Half:
        return 2;
    case torch::ScalarType::Double:
        return 8;
    default:
        return 4;
    }
}

thrust::device_vector<uint8_t>
full_vector(torch::ScalarType scalar_type, float value, size_t count) {
    auto vector =
        thrust::device_vector<uint8_t>(scalar_type_size(scalar_type) * count);

    switch (scalar_type) {
    case torch::ScalarType::Half: {
        const auto half_value = __float2half(value);
        thrust::fill_n(
            thrust::device, reinterpret_cast<__half *>(vector.data().get()),
            count, half_value
        );
    } break;
    case torch::ScalarType::Float: {
        thrust::fill_n(
            thrust::device, reinterpret_cast<float *>(vector.data().get()),
            count, value
        );
    } break;
    case torch::ScalarType::Double: {
        thrust::fill_n(
            thrust::device, reinterpret_cast<double *>(vector.data().get()),
            count, value
        );
    } break;
    }

    return vector;
}

void adjacency_matmul(
    torch::Tensor col_indices, torch::Tensor row_pointers, torch::Tensor matrix,
    torch::Tensor out, bool negate_lhs
) {
    at::cuda::CUDAGuard device_guard(out.device());

    const auto size = matrix.size(0);

    // The cusparse needs a value vectors.
    auto values = full_vector(
        matrix.scalar_type(), negate_lhs ? -1.0 : 1.0, col_indices.numel()
    );

    const auto scalar_type = cusparse_data_type(matrix.scalar_type());
    // FIXME: This is kind of bad. We just assume that matrix is transposed if
    // it isn't contigous.
    const auto rhs_transpose = !matrix.is_contiguous();

    // Create handle for cusparse.
    cusparseHandle_t handle;
    cusparseCreate(&handle);

    // Create cusparse sparse matrix.
    cusparseSpMatDescr_t sparse_desc;
    CHECK_CUSPARSE(cusparseCreateCsr(
        &sparse_desc, size, size, col_indices.size(0), row_pointers.data_ptr(),
        col_indices.data_ptr(), values.data().get(), CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, scalar_type
    ));

    // Create cusparse dense matrices.
    cusparseDnMatDescr_t dense_desc, out_desc;
    CHECK_CUSPARSE(cusparseCreateDnMat(
        &dense_desc, size, size, size, matrix.data_ptr(), scalar_type,
        CUSPARSE_ORDER_ROW
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
    const auto rhs_op = rhs_transpose ? CUSPARSE_OPERATION_TRANSPOSE
                                      : CUSPARSE_OPERATION_NON_TRANSPOSE;
    const auto algo = CUSPARSE_SPMM_ALG_DEFAULT;

    // Determine size of buffer.
    size_t buffer_size;
    CHECK_CUSPARSE(cusparseSpMM_bufferSize(
        handle, lhs_op, rhs_op, &alpha, sparse_desc, dense_desc, &beta,
        out_desc, scalar_type, algo, &buffer_size
    ));

    auto buffer = thrust::device_vector<uint8_t>(buffer_size);

    // Do multiplication.
    CHECK_CUSPARSE(cusparseSpMM(
        handle, lhs_op, rhs_op, &alpha, sparse_desc, dense_desc, &beta,
        out_desc, scalar_type, algo, buffer.data().get()
    ));

    cudaDeviceSynchronize();

    cusparseDestroySpMat(sparse_desc);
    cusparseDestroyDnMat(dense_desc);
    cusparseDestroyDnMat(out_desc);
    cusparseDestroy(handle);
}

//
// Create adjacency.
//

struct Edge {
    int col, row;
};

// Go through `edges` in parallel. When a thread sees something like
// "(0, 0), (1, 0)", it knows where row 1 starts, which it can write to
// `row_pointers` without race conditions.
__global__ void create_row_pointers(
    const Accessor<Edge, 1> edges, Accessor<int, 1> row_pointers
) {
    const auto tid = threadIdx.x + blockIdx.x * blockDim.x;
    const auto stride = blockDim.x * gridDim.x;

    const auto edge_count = edges.size(0);

    // if this is the first thread, the first edge starts at 0. The last edge
    // starts at `edge_count`.
    if (tid == 0) {
        row_pointers[0] = 0;
        row_pointers[row_pointers.size(0) - 1] = edge_count;
    }

    // Go through all edges and check where if there is a change in the row.
    for (auto edge_index = tid; edge_index < edge_count - 1;
         edge_index += stride) {
        const auto current = edges[edge_index];
        const auto next = edges[edge_index + 1];

        // If `current` and `next` have different rows, we know that every row
        // between the two and `next.row` starts on `edge_index + 1`.
        for (auto row_index = current.row + 1; row_index <= next.row;
             row_index++) {
            row_pointers[row_index] = edge_index + 1;
        }
    }
}

// Copy the `col` field of each `Each` in `edges` into `col_indices`.
__global__ void create_col_indices(
    const Accessor<Edge, 1> edges, Accessor<int, 1> col_indices
) {
    const auto tid = threadIdx.x + blockIdx.x * blockDim.x;
    const auto stride = blockDim.x * gridDim.x;

    const auto edge_count = edges.size(0);

    for (auto edge_index = tid; edge_index < edge_count; edge_index += stride) {
        col_indices[edge_index] = edges[edge_index].col;
    }
}

void sort_edges(uint64_t *edges, uint64_t *sorted, long edge_count) {
    // This is only to obtain the required temporary storage size.
    size_t tmp_storage_size = 0;
    cub::DeviceRadixSort::SortKeys(
        nullptr, tmp_storage_size, edges, sorted, edge_count
    );

    // Allocate temporary storage.
    void *tmp_storage = nullptr;
    cudaMalloc(&tmp_storage, tmp_storage_size);

    // Sort edges.
    cub::DeviceRadixSort::SortKeys(
        tmp_storage, tmp_storage_size, edges, sorted, edge_count
    );
    cudaDeviceSynchronize();

    // Deallocate temporary storage.
    cudaFree(tmp_storage);
}

void create_adjacency(
    torch::Tensor edges, torch::Tensor col_indices, torch::Tensor row_pointers
) {
    at::cuda::CUDAGuard device_guard(edges.device());

    constexpr auto block_size = 64;
    constexpr auto thread_count = 1024;

    const auto edge_count = edges.size(0) / 2;
    constexpr auto block_count = div_ceil(thread_count, block_size);

    auto sorted_edges = thrust::device_vector<Edge>(edge_count);

    sort_edges(
        reinterpret_cast<uint64_t *>(edges.data_ptr()),
        reinterpret_cast<uint64_t *>(sorted_edges.data().get()), edge_count
    );

    const auto stride = edges.stride(0);
    const auto edges_accessor =
        Accessor<Edge, 1>(sorted_edges.data().get(), &edge_count, &stride);

    create_row_pointers<<<block_count, block_size>>>(
        edges_accessor, row_pointers.packed_accessor32<int, 1>()
    );

    create_col_indices<<<block_count, block_size>>>(
        edges_accessor, col_indices.packed_accessor32<int, 1>()
    );

    cudaDeviceSynchronize();
}
