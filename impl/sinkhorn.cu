#include <cuda.h>
#include <cuda_runtime.h>

constexpr int block_size = 32;

// Adds `add` to each column of `K` and sums all rows together.
__global__ void kernel(
    const torch::PackedTensorAccessor<float, 2> K,
    const torch::PackedTensorAccessor<float, 1> add,
    torch::PackedTensorAccessor<float, 1> out,
    int axis
) {
    const auto tid = threadIdx.x;
    const auto bid = blockIdx.x;
    const auto size = K.size(axis);

    float max = -INFINITY;
    float sum = 0.0;

    #pragma unroll
    for (int i = tid; i < size; i += block_size) {
        max = fmaxf(max, K[i][bid] + add[i]);
    }

    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        max = fmaxf(max, __shfl_xor_sync(0xffffffff, max, offset));
    }

    #pragma unroll
    for (int i = tid; i < size; i += block_size) {
        sum += expf(K[i][bid] + add[i] - max);
    }

    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        sum += __shfl_xor_sync(0xffffffff, sum, offset);
    }

    if (tid == 0) {
        out[bid] = -(max + logf(sum));
    }
}

torch::Tensor sinkhorn_step_cuda(torch::Tensor K, torch::Tensor add, int axis) {
    const auto blocks = K.size(axis == 0 ? 1 : 0);
    auto out = torch::empty(blocks, K.type()).to(K.device());
    auto K_accessor = K.packed_accessor<float, 2>();

    if (axis == 1) {
        K_accessor = K_accessor.transpose(0, 1);
    }

    kernel<<<blocks, block_size>>>(
        K_accessor,
        add.packed_accessor<float, 1>(),
        out.packed_accessor<float, 1>(),
        axis
    );

    cudaDeviceSynchronize();

    return out;
}