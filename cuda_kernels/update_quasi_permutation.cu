#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/torch.h>
#include <c10/cuda/CUDAGuard.h>
#include "common.cuh"

__global__ void log_kernel(
    Accessor<float, 2> P,
    Accessor<float, 2> K,
    Accessor<float, 1> log_u,
    Accessor<float, 1> log_v,
    Accessor<float, 1> duality_gaps,
    float alpha,
    float sinkhorn_regularization,
    size_t size
) {
    const auto tid = threadIdx.x;
    const auto bid = blockIdx.x;

    float duality_gap = 0.0;

#pragma unroll
    for (auto col = tid; col < size; col += blockDim.x) {
        const auto scaled = __expf(K[bid][col] + log_v[col] + log_u[bid]);
        const auto difference = scaled - P[bid][col];
        duality_gap += difference * (K[bid][col] * -sinkhorn_regularization);
        P[bid][col] += difference * alpha;
    }

    duality_gap = block_sum_reduce(duality_gap);

    if (tid == 0) {
        duality_gaps[bid] = duality_gap;
    }
}

float update_quasi_permutation_log(
    torch::Tensor P,
    torch::Tensor K,
    torch::Tensor u,
    torch::Tensor v,
    float alpha,
    float sinkhorn_regularization
) {
    at::cuda::CUDAGuard device_guard(P.device());
    const auto block_size = 32 * 12;
    torch::Tensor duality_gaps = torch::empty_like(u);
    log_kernel<<<P.size(0), block_size>>>(
        P.packed_accessor32<float, 2>(), K.packed_accessor32<float, 2>(),
        u.packed_accessor32<float, 1>(), v.packed_accessor32<float, 1>(),
        duality_gaps.packed_accessor32<float, 1>(),
        alpha, sinkhorn_regularization, P.size(0)
    );
    cudaDeviceSynchronize();
    return std::abs(duality_gaps.sum().cpu().item().toFloat());
}
