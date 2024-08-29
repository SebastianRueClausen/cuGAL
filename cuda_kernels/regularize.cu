#include "common.cuh"
#include <c10/cuda/CUDAGuard.h>
#include <torch/torch.h>
#include <utility>

__device__ inline std::pair<int, int> index_to_coord(int index, int size) {
    return std::make_pair(index / size, index % size);
}

__global__ void kernel(
    Accessor<float, 2> gradient, Accessor<float, 2> P, int iteration,
    size_t entry_count
) {
    const auto size = gradient.size(0);
    const auto index = threadIdx.x + blockIdx.x * blockDim.x;

    if (index >= entry_count)
        return;

    const auto [x, y] = index_to_coord(index, size);
    gradient[x][y] += iteration - iteration * 2 * P[x][y];
}

void regularize(torch::Tensor gradient, torch::Tensor P, int iteration) {
    at::cuda::CUDAGuard device_guard(gradient.device());

    const auto size = gradient.size(0);
    const auto entry_count = size * size;

    const auto block_size = 256;
    const auto block_count = div_ceil(entry_count, block_size);

    kernel<<<block_count, block_size>>>(
        gradient.packed_accessor32<float, 2>(), P.packed_accessor32<float, 2>(),
        iteration, entry_count
    );

    cudaDeviceSynchronize();
}
