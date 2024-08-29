#include "common.cuh"
#include <c10/cuda/CUDAGuard.h>
#include <iostream>
#include <torch/torch.h>
#include <utility>

__device__ inline std::pair<int, int> index_to_coord(int index, int size) {
    return std::make_pair(index / size, index % size);
}

__global__ void kernel(
    Accessor<float, 2> source, Accessor<float, 2> target,
    Accessor<float, 2> out, size_t entry_count
) {
    const auto size = out.size(0);
    const auto index = threadIdx.x + blockIdx.x * blockDim.x;

    if (index >= entry_count)
        return;

    const auto [x, y] = index_to_coord(index, size);
    const auto a = source[x];
    const auto b = target[y];

    auto sum = 0.0;
#pragma unroll
    for (auto i = 0; i < 4; i++) {
        const auto diff = a[i] - b[i];
        sum += diff * diff;
    }

    out[x][y] += sqrtf(sum);
}

void add_distance(
    torch::Tensor source, torch::Tensor target, torch::Tensor out
) {
    at::cuda::CUDAGuard device_guard(out.device());

    const auto size = out.size(0);
    const auto entry_count = size * size;

    const auto block_size = 256;
    const auto block_count = div_ceil(entry_count, block_size);

    kernel<<<block_count, block_size>>>(
        source.packed_accessor32<float, 2>(),
        target.packed_accessor32<float, 2>(), out.packed_accessor32<float, 2>(),
        entry_count
    );

    cudaDeviceSynchronize();
}
