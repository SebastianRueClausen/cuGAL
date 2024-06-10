#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <torch/torch.h>
#include <c10/cuda/CUDAGuard.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/execution_policy.h>
#include "common.cuh"

__device__ float max_rows(
    const Accessor<float, 2> K,
    const Accessor<float, 1> add,
    const size_t size,
    const unsigned int tid,
    const unsigned int bid)
{
    float max = -INFINITY;

#pragma unroll
    for (auto i = tid; i < size; i += blockDim.x)
    {
        max = fmaxf(max, K[bid][i] + add[i]);
    }

    max = block_max_reduce(max);

    return max;
}

__device__ float max_cols(
    const Accessor<float, 2> K,
    const Accessor<float, 1> add,
    const size_t size,
    const unsigned int tid,
    const unsigned int bid)
{
    float max = -INFINITY;

#pragma unroll
    for (auto i = tid; i < size; i += blockDim.x)
    {
        max = fmaxf(max, K[i][bid] + add[i]);
    }

    max = block_max_reduce(max);

    return max;
}

__device__ float sum_rows(
    const Accessor<float, 2> K,
    const Accessor<float, 1> add,
    const size_t size,
    const unsigned int tid,
    const unsigned int bid,
    const float max)
{
    float sum = 0.0;

#pragma unroll
    for (auto i = tid; i < size; i += blockDim.x)
    {
        sum += expf(K[bid][i] + add[i] - max);
    }

    sum = block_sum_reduce(sum);
    return sum;
}

__device__ float sum_cols(
    const Accessor<float, 2> K,
    const Accessor<float, 1> add,
    const size_t size,
    const unsigned int tid,
    const unsigned int bid,
    const float max)
{
    float sum = 0.0;

#pragma unroll
    for (auto i = tid; i < size; i += blockDim.x)
    {
        sum += expf(K[i][bid] + add[i] - max);
    }

    sum = block_sum_reduce(sum);
    return sum;
}

// Adds `add` to each column of `K` and sums all rows together.
__global__ void kernel_rows(
    const Accessor<float, 2> K,
    const Accessor<float, 1> add,
    Accessor<float, 1> out,
    const size_t size)
{
    const auto tid = threadIdx.x;
    const auto bid = blockIdx.x;

    auto max = max_rows(K, add, size, tid, bid);

    auto sum = sum_rows(K, add, size, tid, bid, max);

    if (tid == 0)
    {
        out[bid] = -(max + logf(sum));
    }
}

__global__ void kernel_cols(
    const Accessor<float, 2> K,
    const Accessor<float, 1> add,
    Accessor<float, 1> out,
    const size_t size)
{
    const auto tid = threadIdx.x;
    const auto bid = blockIdx.x;

    auto max = max_cols(K, add, size, tid, bid);

    auto sum = sum_cols(K, add, size, tid, bid, max);

    if (tid == 0)
    {
        out[bid] = -(max + logf(sum));
    }
}

__global__ void kernel_half2(
    const Accessor<__half2, 2> K,
    const Accessor<__half2, 1> add,
    Accessor<__half, 1> out,
    const size_t size)
{
    const auto tid = threadIdx.x;
    const auto bid = blockIdx.x;

    __half2 max = __float2half2_rn(-INFINITY);
    __half2 sum = __float2half2_rn(0.0);

#pragma unroll
    for (auto i = tid; i < size; i += blockDim.x)
    {
        max = __hmax2(max, __hadd2(K[bid][i], add[i]));
    }

    max = block_max_reduce_half2(max);

    // The lower part of max contains the max of all even entries, while the
    // high part contains the max of the odd.
    const auto joined_max = __hmax(__low2half(max), __high2half(max));
    max = __half2half2(joined_max);

#pragma unroll
    for (auto i = tid; i < size; i += blockDim.x)
    {
        sum = __hadd2(sum, h2exp(__hadd2(K[bid][i], __hsub2(add[i], max))));
    }

    sum = block_sum_reduce_half2(sum);
    const auto joined_sum = __hadd(__low2half(sum), __high2half(sum));

    if (tid == 0)
    {
        out[bid] = __hneg(__hadd(joined_max, hlog(joined_sum)));
    }
}

constexpr size_t block_size = 32 * 12;

void sinkhorn_step_cuda(torch::Tensor K, torch::Tensor add, torch::Tensor out, bool rows)
{
    at::cuda::CUDAGuard device_guard(K.device());

    const auto blocks = K.size(0);

    if (K.scalar_type() == torch::ScalarType::Float)
    {
        if (rows)
        {
            kernel_rows<<<blocks, block_size>>>(
                K.packed_accessor32<float, 2>(),
                add.packed_accessor32<float, 1>(),
                out.packed_accessor32<float, 1>(),
                K.size(1));
        }
        else
        {
            kernel_cols<<<blocks, block_size>>>(
                K.packed_accessor32<float, 2>(),
                add.packed_accessor32<float, 1>(),
                out.packed_accessor32<float, 1>(),
                K.size(0));
        }
    }
    else if (K.scalar_type() == torch::ScalarType::Half)
    {
        const auto K_ptr = reinterpret_cast<__half2 *>(K.data_ptr());
        const int64_t K_sizes[2] = {K.size(0), K.size(1) / 2};
        const int64_t K_strides[2] = {K.stride(0) / 2, K.stride(1)};

        const auto add_ptr = reinterpret_cast<__half2 *>(add.data_ptr());
        const auto add_size = add.size(0) / 2;
        const auto add_stride = add.stride(0);

        const auto out_ptr = reinterpret_cast<__half *>(out.data_ptr());
        const auto out_size = out.size(0);
        const auto out_stride = out.stride(0);

        kernel_half2<<<blocks, block_size>>>(
            Accessor<__half2, 2>(K_ptr, K_sizes, K_strides),
            Accessor<__half2, 1>(add_ptr, &add_size, &add_stride),
            Accessor<__half, 1>(out_ptr, &out_size, &out_stride),
            add_size);
    }
    else
    {
        printf("invalid data type\n");
        abort();
    }

    cudaDeviceSynchronize();
}

// Adds `add` to each column of `K` and sums all rows together.
__global__ void create_max_vectors(
    const Accessor<float, 2> K,
    const Accessor<float, 1> add,
    float *maxtmp,
    const int totalBlocks,
    const size_t size)
{
    const auto tid = threadIdx.x;
    const auto bid = blockIdx.x;

    const auto tmpStart = bid % totalBlocks * size;

#pragma unroll
    for (auto i = tid; i < size; i += blockDim.x)
    {
        maxtmp[tmpStart + i] = fmaxf(K[bid][i], maxtmp[tmpStart + i]);
    }
}

__global__ void reduce_max_vectors(
    float *maxtmp,
    const int totalBlocks,
    const size_t size)
{
    const auto tid = threadIdx.x;
    const auto bid = blockIdx.x;
    const auto gid = bid * blockDim.x + tid;

    if (gid >= size)
        return;

    auto max = -INFINITY;

#pragma unroll
    for (auto i = gid; i < size; i += size)
    {
        maxtmp[i] = fmaxf(maxtmp[i], max);
    }

    maxtmp[gid] = max;
}

__global__ void create_sum_vectors(
    const Accessor<float, 2> K,
    const Accessor<float, 1> add,
    float *maxtmp,
    float *sumtmp,
    const int totalBlocks,
    const size_t size)
{
    const auto tid = threadIdx.x;
    const auto bid = blockIdx.x;

    const auto tmpStart = bid % totalBlocks * size;

#pragma unroll
    for (auto i = tid; i < size; i += blockDim.x)
    {
        sumtmp[tmpStart + i] += expf(K[bid][i] + add[i] - maxtmp[i]);
    }
}

__global__ void reduce_sum_vectors(
    float *maxtmp,
    float *sumtmp,
    Accessor<float, 1> out,
    const size_t size)
{
    const auto tid = threadIdx.x;
    const auto bid = blockIdx.x;
    const auto gid = bid * blockDim.x + tid;

    if (gid >= size)
        return;

    auto sum = 0.0;

#pragma unroll
    for (auto i = gid; i < size; i += size)
    {
        sum += sumtmp[i];
    }

    out[gid] = -(maxtmp[gid] + logf(sum));
}

void sinkhorn_step_cols(
    torch::Tensor K,
    torch::Tensor add,
    torch::Tensor out)
{
    at::cuda::CUDAGuard device_guard(K.device());

    int numSMs;
    int maxBlocksPerSM;

    int deviceIdx = K.device().index();

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceIdx);
    numSMs = deviceProp.multiProcessorCount;

    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxBlocksPerSM, create_max_vectors, block_size, 0);

    printf("Max blocks per SM: %d\n", maxBlocksPerSM);

    int totalBlocks = numSMs * maxBlocksPerSM;

    auto max_vector = thrust::device_vector<float>(K.size(0) * totalBlocks);
    thrust::fill_n(thrust::device, max_vector.data(), max_vector.size(), -INFINITY);

    printf("Total blocks: %d\n", totalBlocks);

    create_max_vectors<<<totalBlocks, block_size>>>(
        K.packed_accessor32<float, 2>(),
        add.packed_accessor32<float, 1>(),
        max_vector.data().get(),
        totalBlocks,
        K.size(1));

    printf("Created max vectors\n");

    cudaDeviceSynchronize();
    // Reduce max vectors
    reduce_max_vectors<<<div_ceil(K.size(0), block_size), block_size>>>(
        max_vector.data().get(),
        totalBlocks,
        K.size(1));

    printf("Reduced max vectors\n");

    cudaDeviceSynchronize();

    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxBlocksPerSM, create_sum_vectors, block_size, 0);

    totalBlocks = numSMs * maxBlocksPerSM;

    auto sum_vector = thrust::device_vector<float>(K.size(0) * totalBlocks);
    thrust::fill_n(thrust::device, sum_vector.data(), sum_vector.size(), 0.0);

    printf("Total blocks: %d\n", totalBlocks);

    create_sum_vectors<<<totalBlocks, block_size>>>(
        K.packed_accessor32<float, 2>(),
        add.packed_accessor32<float, 1>(),
        max_vector.data().get(),
        sum_vector.data().get(),
        totalBlocks,
        K.size(1));

    printf("Created sum vectors\n");

    cudaDeviceSynchronize();

    // Reduce sum vectors
    reduce_sum_vectors<<<div_ceil(K.size(0), block_size), block_size>>>(
        max_vector.data().get(),
        sum_vector.data().get(),
        out.packed_accessor32<float, 1>(),
        K.size(1));

    printf("Reduced sum vectors\n");

    cudaDeviceSynchronize();
}
