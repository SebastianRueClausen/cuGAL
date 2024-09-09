#include "block.cuh"
#include "common.cuh"
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/torch.h>

// Adds `add` to each column of `K` and sums all rows together.
__global__ void kernel(
    const Accessor<float, 2> K, const Accessor<float, 1> add,
    Accessor<float, 1> out, const size_t size
) {
    const auto tid = threadIdx.x;
    const auto bid = blockIdx.x;

    float max = -INFINITY;
    float sum = 0.0;

#pragma unroll
    for (auto i = tid; i < size; i += blockDim.x) {
        max = fmaxf(max, K[bid][i] + add[i]);
    }

    max = block_max_reduce(max);

#pragma unroll
    for (auto i = tid; i < size; i += blockDim.x) {
        sum += expf(K[bid][i] + add[i] - max);
    }

    sum = block_sum_reduce(sum);

    if (tid == 0) {
        out[bid] = -(max + logf(sum));
    }
}

void sinkhorn_log_step(torch::Tensor K, torch::Tensor add, torch::Tensor out) {
    at::cuda::CUDAGuard device_guard(out.device());
    kernel<<<K.size(0), 32 * 12>>>(
        K.packed_accessor32<float, 2>(), add.packed_accessor32<float, 1>(),
        out.packed_accessor32<float, 1>(), K.size(0)
    );
    cudaDeviceSynchronize();
}

void sinkhorn_log_step_stream(
    torch::Tensor K, torch::Tensor add, torch::Tensor out, long rows_per_block
) {
    const long input_sizes[2] = {rows_per_block, K.size(1)};
    const long output_sizes[2] = {1, 1};
    auto blocks = create_max_possible_blocks(
        input_sizes, output_sizes, div_ceil(K.size(0), rows_per_block)
    );
    for (long round = 0, rows_uploaded = 0; rows_uploaded < K.size(0);
         round++, rows_uploaded += rows_per_block) {
        auto block = &blocks[round % blocks.size()];
        const auto row_amount =
            std::min(K.size(0) - rows_uploaded, rows_per_block);
        block->upload(
            K.data_ptr<float>() + rows_uploaded * K.size(1), row_amount
        );
        const auto out_accessor = out.slice(0, rows_uploaded);
        kernel<<<row_amount, 32 * 12, 0, block->stream>>>(
            block->input, add.packed_accessor32<float, 1>(),
            out_accessor.packed_accessor32<float, 1>(), K.size(1)
        );
    }
    cudaDeviceSynchronize();
    for (auto block : blocks)
        block.destroy();
}
