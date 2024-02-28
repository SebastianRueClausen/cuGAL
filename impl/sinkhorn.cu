#include <cuda.h>
#include <cuda_runtime.h>

constexpr int block_size = 64;

template <typename scalar_t>
__device__ __forceinline__
scalar_t maxg(scalar_t a, scalar_t b) {
    return fmaxf(a, b);
}

template <>
__device__ __forceinline__
__half maxg<__half>(__half a, __half b) {
    return __hmax_nan(a, b);
}

template <>
__device__ __forceinline__
double maxg<double>(double a, double b) {
    return fmax(a, b);
}

template <typename scalar_t>
__device__ __forceinline__
scalar_t expg(scalar_t x) {
    return expf(x);
}

template <>
__device__ __forceinline__
__half expg<__half>(__half x) {
    return hexp(x);
}

template <>
__device__ __forceinline__
double expg<double>(double x) {
    return exp(x);
}

// Adds `add` to each column of `K` and sums all rows together.
template <typename scalar_t>
__global__ void kernel(
    const torch::PackedTensorAccessor<scalar_t, 2> K,
    const torch::PackedTensorAccessor<scalar_t, 1> add,
    torch::PackedTensorAccessor<scalar_t, 1> out,
    int axis
) {
    __shared__ scalar_t results[block_size];
    __shared__ scalar_t factor;

    const auto tid = threadIdx.x;
    const auto bid = blockIdx.x;

    const auto size = K.size(axis);

    const auto add_value = add[bid];

    scalar_t max = -INFINITY;
    for (int i = tid; i < size; i += block_size) {
        max = maxg(K[i][bid] + add[i], max);
    }

    results[tid] = max;

    #pragma unroll
    for (int i = block_size / 2; i > 0; i >>= 1) {
        __syncthreads();

        if (tid < i) {
            results[tid] = maxg(results[tid + i], results[tid]);
        }
    }

    if (tid == 0) {
        factor = results[0];
    }

    __syncthreads();

    scalar_t sum = 0.0;
    for (int i = tid; i < size; i += block_size) {
        sum += expg(K[i][bid] + add[i] - factor);
    }

    results[tid] = sum;

    #pragma unroll
    for (int i = block_size / 2; i > 0; i >>= 1) {
        __syncthreads();

        if (tid < i) {
            results[tid] += results[tid + i];
        }
    }

    if (tid == 0) {
        out[bid] = -(factor + logf(results[0]));
    }
}

torch::Tensor sinkhorn_step_cuda(torch::Tensor K, torch::Tensor add, int axis) {
    const auto blocks = K.size(axis == 0 ? 1 : 0);

    auto out = torch::empty(blocks, K.type()).to(K.device());

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(K.type(), "kernel", ([&] {
        auto K_accessor = K.packed_accessor<scalar_t, 2>();

        if (axis == 1) {
            K_accessor = K_accessor.transpose(0, 1);
        }

        kernel<scalar_t><<<blocks, block_size>>>(
            K_accessor,
            add.packed_accessor<scalar_t, 1>(),
            out.packed_accessor<scalar_t, 1>(),
            axis
        );
    }));

    cudaDeviceSynchronize();

    return out;
}