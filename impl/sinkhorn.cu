#include <cuda.h>
#include <cuda_runtime.h>

const int block_size = 64;

template <typename scalar_t>
__global__ void kernel(
    torch::PackedTensorAccessor<scalar_t, 2> x,
    torch::PackedTensorAccessor<scalar_t, 1> out,
    int axis
) {
    __shared__ scalar_t results[block_size];
    __shared__ scalar_t factor;

    auto tid = threadIdx.x;
    auto bid = blockIdx.x;

    auto size = x.size(axis == 0 ? 1 : 0);

    scalar_t max = -INFINITY;
    for (int i = tid; i < size; i += block_size) {
        max = fmaxf(x[bid][i], max);
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
        sum += expf(x[bid][i] - factor);
    }

    results[tid] = sum;

    for (int i = block_size / 2; i > 0; i >>= 1) {
        __syncthreads();

        if (tid < i) {
            results[tid] += results[tid + i];
        }
    }

    if (tid == 0) {
        out[bid] = factor + logf(results[0]);
    }
}

torch::Tensor logsumexp_cuda(torch::Tensor x, int axis) {
    auto blocks = x.size(axis == 0 ? 1 : 0);
    auto threads = block_size;

    auto out = torch::empty(blocks, x.type()).to(x.device());

    kernel<float><<<blocks, threads>>>(x.packed_accessor<float, 2>(), out.packed_accessor<float, 1>(), axis);

    return out;
}