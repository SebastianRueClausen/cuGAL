#include <cuda.h>
#include <cuda_runtime.h>
#include <torch.h>
#include <c10/cuda/CUDAGuard.h>
#include "common.h"

__device__ void log_kernel(
    Accessor<float, 2> P,
    Accessor<float, 2> K,
    Accessor<float, 1> log_u,
    Accessor<float, 1> log_v,
    float alpha,
    size_t size
) {
    const auto tid = threadIdx.x;
    const auto bid = blockIdx.x;
#pragma unroll
    for (auto col = tid; col < size; col += blockDim.x) {
        float scaled = expf(K[bid][col] + log_v[bid] + log_u[col]);
        P[bid][col] += (scaled - P[bid][col]) * alpha;
    }
}

__device__ void kernel
    Accessor<float, 2> P,
    Accessor<float, 2> K,
    Accessor<float, 1> u,
    Accessor<float, 1> v,
    float alpha,
    size_t size
 {
    const auto tid = threadIdx.x;
    const auto bid = blockIdx.x;
#pragma unroll
    for (auto col = tid; col < size; col += blockDim.x) {
        float scaled = K[bid][col] * v[bid] * u[col];
        P[bid][col] += (scaled - P[bid][col]) * alpha;
    }

}

void update_quasi_permutation(
    torch::Tensor P,
    torch::Tensor K,
    torch::Tensor u,
    torch::Tensor v,
    float alpha,
    bool log
) {
    at::cuda::CUDAGuard device_guard(P.device());
    const auto block_size = 32 * 12;
    if (log) {
        log_kernel<<<P.size(0), block_size>>>(
            P.packed_accessor32<float, 2>(), K.packed_accessor32<float, 2>(),
            u.packed_accessor32<float, 1>(), v.packed_accessor32<float, 1>(),
            alpha, P.size(0)
        );
    } else {
        kernel<<<P.size(0), block_size>>>(
            P.packed_accessor32<float, 2>(), K.packed_accessor32<float, 2>(),
            u.packed_accessor32<float, 1>(), v.packed_accessor32<float, 1>(),
            alpha, P.size(0)
        );
    }

    cudaDeviceSynchronize();
}