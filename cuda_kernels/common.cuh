#pragma once

#include <cuda_fp16.h>
#include <torch/torch.h>

template <typename scalar_t, size_t dims>
using Accessor = torch::PackedTensorAccessor32<scalar_t, dims>;

inline Accessor<float, 1> flatten_accessor(Accessor<float, 2> accessor) {
    const long size = accessor.size(0) * accessor.size(1);
    const long stride = 1;
    return Accessor<float, 1>(accessor.data(), &size, &stride);
}

inline constexpr int div_ceil(int x, int y) {
    return (x + y - 1) / y;
}

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

// Performs a max reduction within a single warp.
__device__ inline float warp_min_reduce(float min) {
#pragma unroll
    for (auto offset = warpSize / 2; offset > 0; offset >>= 1) {
        min = fminf(min, __shfl_xor_sync(__activemask(), min, offset));
    }
    return min;
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
template <typename scalar_t>
__device__ inline scalar_t warp_lane_swap(scalar_t value, scalar_t default_value) {
    const auto lane = threadIdx.x % warpSize;
    const auto warp = threadIdx.x / warpSize;

    __shared__ scalar_t shared[32];

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
    return warp_sum_reduce(warp_lane_swap<float>(warp_sum_reduce(value), 0.0));
}

// Performs a max reduction across multiple warps in a block.
__device__ inline float block_max_reduce(float value) {
    return warp_max_reduce(warp_lane_swap<float>(warp_max_reduce(value), -INFINITY));
}

// Performs a min reduction across multiple warps in a block.
__device__ inline float block_min_reduce(float value) {
    return warp_min_reduce(warp_lane_swap<float>(warp_min_reduce(value), INFINITY));
}
