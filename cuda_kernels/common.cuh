#include <cuda_fp16.h>

// Performs a sum reduction within a single warp.
__device__ inline float warp_sum_reduce(float sum) {
    #pragma unroll
    for (auto offset = warpSize / 2; offset > 0; offset >>= 1) {
        sum += __shfl_xor_sync(__activemask(), sum, offset);
    }
    return sum;
}

__device__ inline __half2 warp_sum_reduce_half2(__half2 sum) {
    #pragma unroll
    for (auto offset = warpSize / 2; offset > 0; offset >>= 1) {
        sum = __hadd2(sum, __shfl_xor_sync(__activemask(), sum, offset));
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

__device__ inline __half2 warp_max_reduce_half2(__half2 max) {
    #pragma unroll
    for (auto offset = warpSize / 2; offset > 0; offset >>= 1) {
        max = __hmax2(max, __shfl_xor_sync(__activemask(), max, offset));
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

__device__ inline __half2 block_sum_reduce_half2(__half2 value) {
    const auto default_value = __float2half2_rn(0.0);
    const auto warp_sum = warp_sum_reduce_half2(value);
    return warp_sum_reduce_half2(
        warp_lane_swap<__half2>(warp_sum, default_value)
    );
}

// Performs a max reduction across multiple warps in a block.
__device__ inline float block_max_reduce(float value) {
    return warp_max_reduce(warp_lane_swap<float>(warp_max_reduce(value), -INFINITY));
}

__device__ inline __half2 block_max_reduce_half2(__half2 value) {
    const auto default_value = __float2half2_rn(-INFINITY);
    const auto warp_max = warp_max_reduce_half2(value);
    return warp_max_reduce_half2(
        warp_lane_swap<__half2>(warp_max, default_value)
    );
}