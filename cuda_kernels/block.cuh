#pragma once

#include "common.cuh"
#include <optional>
#include <cuda.h>

inline void check_cuda_error(cudaError_t result) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
}

inline std::optional<Accessor<float, 2>> try_create_accessor(const long sizes[2]) {
    float* memory = nullptr;
    if (cudaMalloc(&memory, sizes[0] * sizes[1] * sizeof(float)) != cudaSuccess) {
        return std::nullopt;
    }
    const long strides[2] = { 1, sizes[0] };
    return std::optional<Accessor<float, 2>> { Accessor<float, 2>(memory, sizes, strides) };
}

struct Block {
    Accessor<float, 2> input;
    Accessor<float, 2> output;
    cudaStream_t stream;

    size_t input_size(size_t row_count) {
        return sizeof(float) * row_count * input.size(1);
    }

    size_t output_size() {
        return sizeof(float) * output.size(0) * output.size(1);
    }

    void upload(float* data, size_t row_count) {
        check_cuda_error(cudaMemcpyAsync(
            input.data(), data, input_size(row_count), cudaMemcpyHostToDevice, stream
        ));
    }

    void download(float* data) {
        check_cuda_error(cudaMemcpyAsync(
            data, output.data(), output_size(), cudaMemcpyDeviceToHost, stream
        ));
    }

    void destroy() {
        check_cuda_error(cudaFree(input.data()));
        check_cuda_error(cudaFree(output.data()));
        check_cuda_error(cudaStreamDestroy(stream));
    }
};

inline std::optional<Block> try_create_block(const long input_sizes[2], const long output_sizes[2]) {
    auto input = try_create_accessor(input_sizes);
    if (!input.has_value()) return std::nullopt;
    auto output = try_create_accessor(output_sizes);
    if (!output.has_value()) {
        check_cuda_error(cudaFree(input.value().data()));
        return std::nullopt;
    }
    cudaStream_t stream;
    check_cuda_error(cudaStreamCreate(&stream));
    return std::optional<Block> { Block { input.value(), output.value(), stream } };
}

inline std::vector<Block> create_max_possible_blocks(
    const long input_sizes[2], const long output_sizes[2], long max_block_count
) {
    std::vector<Block> blocks = {};
    while (blocks.size() < max_block_count) {
        auto block = try_create_block(input_sizes, output_sizes);
        if (!block.has_value()) break;
        blocks.push_back(block.value());
    }
    return blocks;
}
