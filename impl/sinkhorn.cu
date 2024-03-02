#include <cuda.h>
#include <cuda_runtime.h>

// Performs a sum reduction within a single warp.
__device__ inline float warp_sum_reduce(float sum) {
    #pragma unroll
    for (auto offset = warpSize / 2; offset > 0; offset >>= 1) {
        sum += __shfl_xor_sync(__activemask(), sum, offset);
    }
    return sum;
}

// Performs a max reduction within a single warp.
__device__ inline float warp_max_reduce(float max) {
    #pragma unroll
    for (auto offset = warpSize / 2; offset > 0; offset >>= 1) {
        max = fmaxf(max, __shfl_xor_sync(__activemask(), max, offset));
    }
    return max;
}

// | w/l | 0 | 1 | 2 |
// |-----|---|---|---|
// | 0   | a | a | a |
// | 1   | b | b | b |
// | 2   | c | c | c |
//
//         |
//         v
//
// | w/l | 0 | 1 | 2 |
// |-----|---|---|---|
// | 0   | a | b | c |
// | 1   | a | b | c |
// | 2   | a | b | c |
__device__ inline float warp_lane_swap(float value, float default_value) {
	const auto lane = threadIdx.x % warpSize;
	const auto warp = threadIdx.x / warpSize;

	__shared__ float shared[32];

    if (lane == 0) {
	    shared[warp] = value;
    }

	__syncthreads();

    if (lane < blockDim.x / warpSize) {
        return shared[lane];
    } else {
        return default_value;
    }
}

// Performs a max reduction across multiple warps in a block.
__device__ inline float block_sum_reduce(float value) {
    return warp_sum_reduce(warp_lane_swap(warp_sum_reduce(value), 0.0));
}

// Performs a max reduction across multiple warps in a block.
__device__ inline float block_max_reduce(float value) {
    return warp_max_reduce(warp_lane_swap(warp_max_reduce(value), -INFINITY));
}

// Adds `add` to each column of `K` and sums all rows together.
template <size_t block_size>
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
    for (auto i = tid; i < size; i += blockDim.x) {
        max = fmaxf(max, K[i][bid] + add[i]);
    }

    if (block_size > 32) {
        max = block_max_reduce(max);
    } else {
        max = warp_max_reduce(max);
    }

    #pragma unroll
    for (auto i = tid; i < size; i += blockDim.x) {
        sum += expf(K[i][bid] + add[i] - max);
    }

    if (block_size > 32) {
        sum = block_sum_reduce(sum);
    } else {
        sum = warp_sum_reduce(sum);
    }

    if (tid == 0) {
        out[bid] = -(max + logf(sum));
    }
}

constexpr size_t block_size = 64;

void sinkhorn_step_cuda(torch::Tensor K, torch::Tensor add, torch::Tensor out, int axis) {
    const auto blocks = K.size(axis == 0 ? 1 : 0);
    auto K_accessor = K.packed_accessor<float, 2>();

    if (axis == 1) {
        K_accessor = K_accessor.transpose(0, 1);
    }

    kernel<block_size><<<blocks, block_size>>>(
        K_accessor,
        add.packed_accessor<float, 1>(),
        out.packed_accessor<float, 1>(),
        axis
    );

    cudaDeviceSynchronize();
}