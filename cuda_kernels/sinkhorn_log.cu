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
        sum += __expf(K[bid][i] + add[i] - max);
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
