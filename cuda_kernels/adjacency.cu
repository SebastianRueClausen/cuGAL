#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <torch/torch.h>
#include "common.cuh"

__global__ void kernel(
    const torch::PackedTensorAccessor32<int, 1> col_indices,
    const torch::PackedTensorAccessor32<int, 1> row_pointers,
    const torch::PackedTensorAccessor32<float, 2> matrix,
    torch::PackedTensorAccessor32<float, 2> out,
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

    kernel<<<blocks, block_size>>>(
        col_indices.packed_accessor32<int, 1>(),
        row_pointers.packed_accessor32<int, 1>(),
        matrix.packed_accessor32<float, 2>(),
        out.packed_accessor32<float, 2>(),
        size,
        sign
    );

    cudaDeviceSynchronize();
}