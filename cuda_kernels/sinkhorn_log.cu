#include "common.cuh"
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/torch.h>

// Adds `add` to each column of `K` and sums all rows together.
__global__ void kernel_momentum(
    const Accessor<float, 2> K, const Accessor<float, 1> add,
    Accessor<float, 1> out, const size_t size, const float momentum
) {
    const auto tid = threadIdx.x;
    const auto bid = blockIdx.x;

    float max = -INFINITY;
    float sum = 0.0;

#pragma unroll(8)
    for (auto i = tid; i < size; i += blockDim.x) {
        max = fmaxf(max, K[bid][i] + add[i]);
    }

    max = block_max_reduce(max);

#pragma unroll(8)
    for (auto i = tid; i < size; i += blockDim.x) {
        sum += __expf(K[bid][i] + add[i] - max);
    }

    sum = block_sum_reduce(sum);

    if (tid == 0) {
        const auto next = max + logf(sum);
        out[bid] = (1.0 - momentum) * out[bid] - momentum * next;
    }
}

// Adds `add` to each column of `K` and sums all rows together.
__global__ void kernel(
    const Accessor<float, 2> K, const Accessor<float, 1> add,
    Accessor<float, 1> out, const size_t size
) {
    const auto tid = threadIdx.x;
    const auto bid = blockIdx.x;

    float max = -INFINITY;
    float sum = 0.0;

#pragma unroll(8)
    for (auto i = tid; i < size; i += blockDim.x) {
        max = fmaxf(max, K[bid][i] + add[i]);
    }

    max = block_max_reduce(max);

#pragma unroll(8)
    for (auto i = tid; i < size; i += blockDim.x) {
        sum += __expf(K[bid][i] + add[i] - max);
    }

    sum = block_sum_reduce(sum);

    if (tid == 0) {
        out[bid] = -(max + logf(sum));
    }
}

__global__ void calculate_marginal_log(
    Accessor<float, 2> K, Accessor<float, 1> u, Accessor<float, 1> v, size_t size, Accessor<float, 1> out
) {
    const auto tid = threadIdx.x;
    const auto bid = blockIdx.x;

    float sum = 0.0;

#pragma unroll
    for (auto col = tid; col < size; col += blockDim.x) {
        sum += __expf(K[bid][col] + v[col] + u[bid]);
    }

    sum = block_sum_reduce(sum);

    if (tid == 0) {
        out[bid] = std::abs(sum - 1.0);
    }
}

float sinkhorn_log_marginal(torch::Tensor K, torch::Tensor u, torch::Tensor v) {
    at::cuda::CUDAGuard device_guard(K.device());
    auto out = torch::empty_like(u);
    calculate_marginal_log<<<K.size(0), 32 * 12>>>(
        K.packed_accessor32<float, 2>(), u.packed_accessor32<float, 1>(),
        v.packed_accessor32<float, 1>(), K.size(0),
        out.packed_accessor32<float, 1>()
    );
    cudaDeviceSynchronize();
    return out.sum().cpu().item().toFloat();
}

void sinkhorn_log_step(torch::Tensor K, torch::Tensor add, torch::Tensor out, float momentum) {
    at::cuda::CUDAGuard device_guard(out.device());
    if (std::abs(momentum - 1.0) < 1e-4) {
        kernel<<<K.size(0), 32 * 12>>>(
            K.packed_accessor32<float, 2>(), add.packed_accessor32<float, 1>(),
            out.packed_accessor32<float, 1>(), K.size(0)
        );
    } else {
        kernel_momentum<<<K.size(0), 32 * 12>>>(
            K.packed_accessor32<float, 2>(), add.packed_accessor32<float, 1>(),
            out.packed_accessor32<float, 1>(), K.size(0), momentum
        );
    }
    cudaDeviceSynchronize();
}
