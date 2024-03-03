#include <cuda.h>
#include <cuda_runtime.h>
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
    return warp_sum_reduce_half2(warp_lane_swap<__half2>(
        warp_sum_reduce_half2(value),
        default_value
    ));
}

// Performs a max reduction across multiple warps in a block.
__device__ inline float block_max_reduce(float value) {
    return warp_max_reduce(warp_lane_swap<float>(warp_max_reduce(value), -INFINITY));
}

__device__ inline __half2 block_max_reduce_half2(__half2 value) {
    const auto default_value = __float2half2_rn(-INFINITY);
    return warp_max_reduce_half2(warp_lane_swap<__half2>(
        warp_max_reduce_half2(value),
        default_value
    ));
}

// Adds `add` to each column of `K` and sums all rows together.
template <size_t block_size>
__global__ void kernel(
    const torch::PackedTensorAccessor<float, 2> K,
    const torch::PackedTensorAccessor<float, 1> add,
    torch::PackedTensorAccessor<float, 1> out,
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

    if (block_size > warpSize) {
        max = block_max_reduce(max);
    } else {
        max = warp_max_reduce(max);
    }

    #pragma unroll
    for (auto i = tid; i < size; i += blockDim.x) {
        sum += expf(K[bid][i] + add[i] - max);
    }

    if (block_size > warpSize) {
        sum = block_sum_reduce(sum);
    } else {
        sum = warp_sum_reduce(sum);
    }

    if (tid == 0) {
        out[bid] = -(max + logf(sum));
    }
}

template <size_t block_size>
__global__ void kernel_half2(
    const torch::PackedTensorAccessor<__half2, 2> K,
    const torch::PackedTensorAccessor<__half2, 1> add,
    torch::PackedTensorAccessor<__half, 1> out,
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

    if (block_size > warpSize) {
        max = block_max_reduce_half2(max);
    } else {
        max = warp_max_reduce_half2(max);
    }

    // The lower part of max contains the max of all even entries, while the
    // high part contains the max of the odd.
    const auto joined_max = __hmax(__high2half(max), __low2half(max));
    max = __half2half2(joined_max);

    #pragma unroll
    for (auto i = tid; i < size; i += blockDim.x) {
        sum = __hadd2(sum, h2exp(__hsub2(__hadd2(K[bid][i], add[i]), max)));
    }

    if (block_size > warpSize) {
        sum = block_sum_reduce_half2(sum);
    } else {
        sum = warp_sum_reduce_half2(sum);
    }
    
    const auto joined_sum = __hadd(__high2half(sum), __low2half(sum));

    if (tid == 0) {
        out[bid] = __hneg(__hadd(joined_max, hlog(joined_sum)));
    }
}

constexpr size_t block_size = 64;

torch::PackedTensorAccessor<__half2, 1> half2_1d_accessor(torch::Tensor &tensor) {
    const auto size = tensor.size(0) / 2;
    const auto stride = tensor.stride(0);

    return torch::PackedTensorAccessor<__half2, 1>(
        reinterpret_cast<__half2*>(tensor.data_ptr()),
        &size,
        &stride
    );
}

void sinkhorn_step_cuda(torch::Tensor K, torch::Tensor add, torch::Tensor out, int axis) {
    const auto blocks = K.size(axis);

    if (K.scalar_type() == torch::ScalarType::Float) {
        auto K_accessor = K.packed_accessor<float, 2>();

        if (axis == 1) {
            K_accessor = K_accessor.transpose(0, 1);
        }

        kernel<block_size><<<blocks, block_size>>>(
            K_accessor,
            add.packed_accessor<float, 1>(),
            out.packed_accessor<float, 1>(),
            K_accessor.size(axis)
        );
    } else if (K.scalar_type() == torch::ScalarType::Half) {
        const auto K_ptr = reinterpret_cast<__half2*>(K.data_ptr());
        const int64_t K_sizes[2] = { K.size(0), K.size(1) / 2 };
        const int64_t K_strides[2] = { K.stride(0) / 2, K.stride(1) };

        auto K_accessor = torch::PackedTensorAccessor<__half2, 2>(K_ptr, K_sizes, K_strides);

        if (axis != 0) {
            printf("f16 can't be transposed\n");
            abort();
        }

        const auto out_ptr = reinterpret_cast<__half*>(out.data_ptr());
        const auto out_size = out.size(0);
        const auto out_stride = out.stride(0);

        auto out_accessor = torch::PackedTensorAccessor<__half, 1>(out_ptr, &out_size, &out_stride);

        const auto add_ptr = reinterpret_cast<__half2*>(add.data_ptr());
        const auto add_size = add.size(0) / 2;
        const auto add_stride = add.stride(0);

        auto add_accessor = torch::PackedTensorAccessor<__half2, 1>(add_ptr, &add_size, &add_stride);

        kernel_half2<block_size><<<blocks, block_size>>>(K_accessor, add_accessor, out_accessor, add_size);
    } else {
        printf("invalid data type\n");
        abort();
    }

    cudaDeviceSynchronize();
}