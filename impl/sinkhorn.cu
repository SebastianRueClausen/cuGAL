#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void logsumexp_kernel(torch::PackedTensorAccessor<scalar_t, 2> x, torch::PackedTensorAccessor<scalar_t, 1> out, int axis) {
    auto row = threadIdx.x;
    auto cols = x.size(axis == 0 ? 1 : 0);

    scalar_t max_value = -INFINITY;

    for (int col = 0; col < cols; col++) {
        max_value = fmaxf(x[row][col], max_value);
    }

    scalar_t sum = 0.0;

    for (int col = 0; col < cols; col++) {
        sum += expf(x[row][col] - max_value);
    }

    out[row] = max_value + logf(sum);
}

torch::Tensor logsumexp_cuda(torch::Tensor x, int axis) {
    auto blocks = 1;
    auto threads = x.size(axis);

    auto out = torch::empty(threads, x.type()).to(x.device());

    logsumexp_kernel<float><<<blocks, threads>>>(x.packed_accessor<float, 2>(), out.packed_accessor<float, 1>(), axis);

    return out;
}