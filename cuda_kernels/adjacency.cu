#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <torch/torch.h>
#include <cub/cub.cuh>
#include <iostream>
#include "common.cuh"
#include <cusparse.h>

#define CHECK_CUSPARSE(func) { \
    cusparseStatus_t status = (func); \
    if (status != CUSPARSE_STATUS_SUCCESS) { \
        printf("cusparse failed at line %d with error: %s (%d)\n", \
               __LINE__, cusparseGetErrorString(status), status); \
    } \
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

void adjacency_matmul(
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
    const auto rhs_op = rhs_transpose
        ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE;
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

void calculate_gradient_symmetric(
    torch::Tensor A_col_indices,
    torch::Tensor A_row_pointers,
    torch::Tensor B_col_indices,
    torch::Tensor B_row_pointers,
    torch::Tensor P,
    torch::Tensor K,
    torch::Tensor out,
    int iteration
) {
    const auto size = K.size(0);

    cusparseHandle_t handle = nullptr;
    cusparseCreate(&handle);

    const auto max_cols = std::max(A_col_indices.size(0), B_col_indices.size(0));
    const auto options = torch::TensorOptions()
        .dtype(K.dtype())
        .device(K.device());
    const auto ones = torch::ones({max_cols}, options);

    const auto temp = torch::empty_like(out);

    const auto scalar_type = cusparse_data_type(K.scalar_type());
    const auto index_type = CUSPARSE_INDEX_32I;
    const auto index_base = CUSPARSE_INDEX_BASE_ZERO;

    // Create sparse matrices.
    cusparseSpMatDescr_t A_desc, B_desc;
    CHECK_CUSPARSE(cusparseCreateCsr(
        &A_desc, size, size, A_col_indices.size(0), A_row_pointers.data_ptr(), A_col_indices.data_ptr(), ones.data_ptr(),
        index_type, index_type, index_base, scalar_type
    ));
    CHECK_CUSPARSE(cusparseCreateCsr(
        &B_desc, size, size, B_col_indices.size(0), B_row_pointers.data_ptr(), B_col_indices.data_ptr(), ones.data_ptr(),
        index_type, index_type, index_base, scalar_type
    ));

    // Create cusparse dense matrices.
    cusparseDnMatDescr_t P_desc, out_desc, temp_desc;
    CHECK_CUSPARSE(cusparseCreateDnMat(
        &P_desc, size, size, size, P.data_ptr(), scalar_type, CUSPARSE_ORDER_ROW
    ));
    CHECK_CUSPARSE(cusparseCreateDnMat(
        &out_desc, size, size, size, out.data_ptr(), scalar_type, CUSPARSE_ORDER_COL
    ));
    CHECK_CUSPARSE(cusparseCreateDnMat(
        &temp_desc, size, size, size, temp.data_ptr(), scalar_type, CUSPARSE_ORDER_ROW
    ));

    const auto no_transpose = CUSPARSE_OPERATION_NON_TRANSPOSE;
    const auto transpose = CUSPARSE_OPERATION_TRANSPOSE;
    const auto algo = CUSPARSE_SPMM_ALG_DEFAULT;

    const auto alpha = 1.0f;
    const auto beta = 0.0f;

    size_t buffer_size1, buffer_size2;

    // temp = (A @ P).T
    CHECK_CUSPARSE(cusparseSpMM_bufferSize(
        handle, no_transpose, no_transpose, &alpha, A_desc, P_desc, &beta, temp_desc, scalar_type, algo, &buffer_size1
    ));

    // out = (B @ temp).T
    CHECK_CUSPARSE(cusparseSpMM_bufferSize(
        handle, no_transpose, transpose, &alpha, A_desc, P_desc, &beta, out_desc, scalar_type, algo, &buffer_size2
    ));

    const auto buffer_size = std::max(buffer_size1, buffer_size2);

    void* buffer = nullptr;
    cudaMalloc(&buffer, buffer_size);

    // temp = (A @ P).T
    CHECK_CUSPARSE(cusparseSpMM(
        handle, no_transpose, no_transpose, &alpha, A_desc, P_desc, &beta, temp_desc, scalar_type, algo, buffer
    ));

    cudaDeviceSynchronize();

    // out = (B @ temp).T
    CHECK_CUSPARSE(cusparseSpMM(
        handle, no_transpose, transpose, &alpha, B_desc, temp_desc, &beta, out_desc, scalar_type, algo, buffer
    ));

    cudaDeviceSynchronize();

    cusparseDestroySpMat(A_desc);
    cusparseDestroySpMat(B_desc);
    cusparseDestroyDnMat(temp_desc);
    cusparseDestroyDnMat(out_desc);
    cusparseDestroyDnMat(P_desc);
    cusparseDestroy(handle);

    cudaFree(buffer);

    out *= -2;
    out += K;
    out += (iteration - 2 * iteration * P);
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
__global__ void create_row_pointers(const Accessor<Edge, 1> edges, Accessor<int, 1> row_pointers) {
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
__global__ void create_col_indices(const Accessor<Edge, 1> edges, Accessor<int, 1> col_indices) {
    const auto tid = threadIdx.x + blockIdx.x * blockDim.x;
    const auto stride = blockDim.x * gridDim.x;

    const auto size = edges.size(0);

    for (auto edge_index = tid; edge_index < size; edge_index += stride) {
        col_indices[edge_index] = edges[edge_index].col;
    }
}

void sort_edges(uint64_t* edges, uint64_t* sorted, long edge_count) {
    // This is only to obtain the required temporary storage size.
    size_t tmp_storage_size = 0;
    cub::DeviceRadixSort::SortKeys(nullptr, tmp_storage_size, edges, sorted, edge_count);

    // Allocate temporary storage.
    void* tmp_storage = nullptr;
    cudaMalloc(&tmp_storage, tmp_storage_size);

    // Sort edges.
    cub::DeviceRadixSort::SortKeys(tmp_storage, tmp_storage_size, edges, sorted, edge_count);
    cudaDeviceSynchronize();

    // Deallocate temporary storage.
    cudaFree(tmp_storage);
}

void create_adjacency_cuda(torch::Tensor edges, torch::Tensor col_indices, torch::Tensor row_pointers) {
    constexpr auto block_size = 64;
    constexpr auto thread_count = 1024;

    const auto edge_count = edges.size(0) / 2;
    constexpr auto block_count = div_ceil(thread_count, block_size);

    void* sorted_edges = nullptr;
    cudaMalloc(&sorted_edges, edges.numel() * 4);
    const auto sorted_edge_ptr = reinterpret_cast<Edge*>(sorted_edges);

    sort_edges((uint64_t*)edges.data_ptr(), (uint64_t*)sorted_edge_ptr, edge_count);

    const auto stride = edges.stride(0);
    const auto edges_accessor = Accessor<Edge, 1>(sorted_edge_ptr, &edge_count, &stride);

    create_row_pointers<<<block_count, block_size>>>(
        edges_accessor, row_pointers.packed_accessor32<int, 1>()
    );
    create_col_indices<<<block_count, block_size>>>(
        edges_accessor, col_indices.packed_accessor32<int, 1>()
    );

    cudaDeviceSynchronize();
}
