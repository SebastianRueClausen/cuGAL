#include <torch/torch.h>
#include <c10/cuda/CUDAGuard.h>
#include "common.cuh"

__global__ void kernel(Accessor<float, 2> source, Accessor<float, 2> target, Accessor<float, 2> out, float mu)
{
    const auto size = out.size(0);

    const auto x = threadIdx.x + blockIdx.x * blockDim.x;
    const auto y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= size || y >= size)
    {
        return;
    }

    const auto a = source[x];
    const auto b = target[y];

    auto sum = 0.0;

#pragma unroll
    for (auto i = 0; i < 5; i++)
    {
        const auto diff = a[i] - b[i];
        sum = fma(diff, diff, sum);
    }

    out[x][y] += sqrtf(sum) * mu;
}

void add_distance(torch::Tensor source, torch::Tensor target, torch::Tensor out, float mu)
{
    at::cuda::CUDAGuard device_guard(out.device());

    const dim3 block_size(32, 32, 1);
    const dim3 block_count(
        div_ceil(out.size(0), 32), // block_size),
        div_ceil(out.size(1), 32), // block_size),
        1);

    kernel<<<block_count, block_size>>>(
        source.packed_accessor32<float, 2>(),
        target.packed_accessor32<float, 2>(),
        out.packed_accessor32<float, 2>(),
        mu);

    cudaDeviceSynchronize();
}