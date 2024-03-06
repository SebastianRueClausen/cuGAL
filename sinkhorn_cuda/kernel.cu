#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <torch/torch.h>

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

// Adds `add` to each column of `K` and sums all rows together.
__global__ void kernel(
    const torch::PackedTensorAccessor32<float, 2> K,
    const torch::PackedTensorAccessor32<float, 1> add,
    torch::PackedTensorAccessor32<float, 1> out,
    const size_t size
) {
    const auto tid = threadIdx.x;
    const auto bid = blockIdx.x;

    float max = -INFINITY;
    float sum = 0.0;

    #pragma unroll
    for (auto i = tid; i < size; i += blockDim.x) {
        max = fmaxf(max, K[bid][i] + add[i]);
    }

    max = block_max_reduce(max);

    #pragma unroll
    for (auto i = tid; i < size; i += blockDim.x) {
        sum += expf(K[bid][i] + add[i] - max);
    }

    sum = block_sum_reduce(sum);

    if (tid == 0) {
        out[bid] = -(max + logf(sum));
    }
}

__global__ void kernel_half2(
    const torch::PackedTensorAccessor32<__half2, 2> K,
    const torch::PackedTensorAccessor32<__half2, 1> add,
    torch::PackedTensorAccessor32<__half, 1> out,
    const size_t size
) {
    const auto tid = threadIdx.x;
    const auto bid = blockIdx.x;

    __half2 max = __float2half2_rn(-INFINITY);
    __half2 sum = __float2half2_rn(0.0);

    #pragma unroll
    for (auto i = tid; i < size; i += blockDim.x) {
        max = __hmax2(max, __hadd2(K[bid][i], add[i]));
    }

    max = block_max_reduce_half2(max);

    // The lower part of max contains the max of all even entries, while the
    // high part contains the max of the odd.
    const auto joined_max = __hmax(__low2half(max), __high2half(max));
    max = __half2half2(joined_max);

    #pragma unroll
    for (auto i = tid; i < size; i += blockDim.x) {
        sum = __hadd2(sum, h2exp(__hadd2(K[bid][i], __hsub2(add[i], max))));
    }

    sum = block_sum_reduce_half2(sum);
    const auto joined_sum = __hadd(__low2half(sum), __high2half(sum));

    if (tid == 0) {
        out[bid] = __hneg(__hadd(joined_max, hlog(joined_sum)));
    }
}

constexpr size_t block_size = 32 * 12;

void sinkhorn_step_cuda(torch::Tensor K, torch::Tensor add, torch::Tensor out) {
    const auto blocks = K.size(0);

    if (K.scalar_type() == torch::ScalarType::Float) {
        kernel<<<blocks, block_size>>>(
            K.packed_accessor32<float, 2>(),
            add.packed_accessor32<float, 1>(),
            out.packed_accessor32<float, 1>(),
            K.size(0)
        );
    } else if (K.scalar_type() == torch::ScalarType::Half) {
        const auto K_ptr = reinterpret_cast<__half2*>(K.data_ptr());
        const int64_t K_sizes[2] = { K.size(0), K.size(1) / 2 };
        const int64_t K_strides[2] = { K.stride(0) / 2, K.stride(1) };

        const auto add_ptr = reinterpret_cast<__half2*>(add.data_ptr());
        const auto add_size = add.size(0) / 2;
        const auto add_stride = add.stride(0);

        const auto out_ptr = reinterpret_cast<__half*>(out.data_ptr());
        const auto out_size = out.size(0);
        const auto out_stride = out.stride(0);

        kernel_half2<<<blocks, block_size>>>(
            torch::PackedTensorAccessor32<__half2, 2>(K_ptr, K_sizes, K_strides),
            torch::PackedTensorAccessor32<__half2, 1>(add_ptr, &add_size, &add_stride),
            torch::PackedTensorAccessor32<__half, 1>(out_ptr, &out_size, &out_stride),
            add_size
        );
    } else {
        printf("invalid data type\n");
        abort();
    }

    cudaDeviceSynchronize();
}