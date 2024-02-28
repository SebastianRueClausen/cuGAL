#include <cuda.h>
#include <cuda_runtime.h>

const int block_size = 64;

template <typename scalar_t>
__global__ void kernel(
    torch::PackedTensorAccessor<scalar_t, 2> K,
    torch::PackedTensorAccessor<scalar_t, 1> add,
    torch::PackedTensorAccessor<scalar_t, 1> out,
    int axis
) {
    __shared__ scalar_t results[block_size];
    __shared__ scalar_t factor;

    auto tid = threadIdx.x;
    auto bid = blockIdx.x;

    auto size = K.size(axis == 0 ? 1 : 0);

    scalar_t max = -INFINITY;
    for (int i = tid; i < size; i += block_size) {
        max = fmaxf(K[bid][i] + add[i], max);
    }

    results[tid] = max;

    for (int i = block_size / 2; i > 0; i >>= 1) {
        __syncthreads();

        if (tid < i) {
            results[tid] = fmaxf(results[tid + i], results[tid]);
        }
    }

    if (tid == 0) {
        factor = results[0];
    }

    __syncthreads();

    scalar_t sum = 0.0;
    for (int i = tid; i < size; i += block_size) {
        sum += expf(K[bid][i] + add[i] - factor);
    }

    results[tid] = sum;

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
    auto blocks = K.size(axis == 0 ? 1 : 0);
    auto threads = block_size;

    auto out = torch::empty(blocks, K.type()).to(K.device());

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(K.type(), "kernel", ([&] {
        kernel<scalar_t><<<blocks, threads>>>(
            K.packed_accessor<scalar_t, 2>(),
            add.packed_accessor<scalar_t, 1>(),
            out.packed_accessor<scalar_t, 1>(),
            axis
        );
    }));

    return out;
}