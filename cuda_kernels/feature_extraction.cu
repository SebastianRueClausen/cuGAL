#include <torch/torch.h>
#include <utility>
#include "common.cuh"

__device__ std::pair<int, int> edge_range(
    uint32_t edge_count,
    const Accessor<int, 1> row_pointers,
    uint32_t vertex_index
) {
    return std::make_pair(
        row_pointers[vertex_index],
        row_pointers.size(0) == vertex_index + 1 ? edge_count : row_pointers[vertex_index + 1]
    );
}

template <typename index_t>
__device__ int intersection_size(
    const Accessor<index_t, 1> col_indices,
    const Accessor<int, 1> row_pointers,
    uint32_t vertex_a,
    uint32_t vertex_b
) {
    const auto [a_start, a_end] = edge_range(col_indices.size(0), row_pointers, vertex_a);
    const auto [b_start, b_end] = edge_range(col_indices.size(0), row_pointers, vertex_b);

    int intersection_size = 0;

    for (auto a_index = a_start, b_index = b_start; a_index < a_end && b_index < b_end;) {
        const auto a_edge = col_indices[a_index];
        const auto b_edge = col_indices[b_index];

        if (a_edge == b_edge) {
            intersection_size++;
            a_index++;
            b_index++;
        } else if (a_edge > b_edge) {
            b_index++;
        } else {
            a_index++;
        }
    }

    return intersection_size;
}

template <typename index_t>
__global__ void vertex_features(
    const Accessor<index_t, 1> col_indices,
    const Accessor<int, 1> row_pointers,
    Accessor<float, 1> clustering,
    Accessor<float, 1> degrees
) {
    const auto vertex_index = threadIdx.x + blockIdx.x * blockDim.x;

    if (vertex_index >= row_pointers.size(0)) {
        return;
    }

    auto triangle_count = 0;
    const auto [start, end] = edge_range(col_indices.size(0), row_pointers, vertex_index);
    for (auto edge_index = start; edge_index < end; edge_index++) {
        triangle_count += intersection_size(col_indices, row_pointers, vertex_index, col_indices[edge_index]);
    }

    const auto degree = end - start;
    degrees[vertex_index]  = float(degree);
    clustering[vertex_index] = triangle_count == 0 ? 0.0 : float(triangle_count) / (degree * (degree - 1));
}

void graph_features(torch::Tensor col_indices, torch::Tensor row_pointers, torch::Tensor clustering, torch::Tensor degrees) {
    constexpr auto block_size = 64;

    const auto thread_count = row_pointers.size(0);
    const auto block_count = div_ceil(thread_count, block_size);

    if (col_indices.scalar_type() == torch::ScalarType::Short) {
        vertex_features<short><<<block_count, block_size>>>(
            col_indices.packed_accessor32<short, 1>(),
            row_pointers.packed_accessor32<int, 1>(),
            clustering.packed_accessor32<float,  1>(),
            degrees.packed_accessor32<float,  1>()
        );
    } else if (col_indices.scalar_type() == torch::ScalarType::Int) {
        vertex_features<int><<<block_count, block_size>>>(
            col_indices.packed_accessor32<int, 1>(),
            row_pointers.packed_accessor32<int, 1>(),
            clustering.packed_accessor32<float,  1>(),
            degrees.packed_accessor32<float,  1>()
        );
    }

    cudaDeviceSynchronize();
}

template <typename index_t>
__global__ void vertex_average_neighbor_features(
    const Accessor<index_t, 1> col_indices,
    const Accessor<int, 1> row_pointers,
    const Accessor<float, 1> features,
    Accessor<float, 1> averages
) {
    const auto vertex_index = threadIdx.x + blockIdx.x * blockDim.x;

    if (vertex_index >= row_pointers.size(0)) {
        return;
    }

    const auto [start, end] = edge_range(col_indices.size(0), row_pointers, vertex_index);

    float sum = 0.0;
    for (auto edge_index = start; edge_index < end; edge_index++) {
        sum += features[col_indices[edge_index]];
    }

    const auto degree = end - start;
    averages[vertex_index] = degree == 0 ? 0.0 : sum / degree;
}

void average_neighbor_features(
    torch::Tensor col_indices,
    torch::Tensor row_pointers,
    torch::Tensor features,
    torch::Tensor averages
) {
    constexpr auto block_size = 64;

    const auto thread_count = row_pointers.size(0);
    const auto block_count = div_ceil(thread_count, block_size);

    if (col_indices.scalar_type() == torch::ScalarType::Short) {
        vertex_average_neighbor_features<short><<<block_count, block_size>>>(
            col_indices.packed_accessor32<short, 1>(),
            row_pointers.packed_accessor32<int, 1>(),
            features.packed_accessor32<float,  1>(),
            averages.packed_accessor32<float,  1>()
        );
    } else if (col_indices.scalar_type() == torch::ScalarType::Int) {
        vertex_average_neighbor_features<int><<<block_count, block_size>>>(
            col_indices.packed_accessor32<int, 1>(),
            row_pointers.packed_accessor32<int, 1>(),
            features.packed_accessor32<float,  1>(),
            averages.packed_accessor32<float,  1>()
        );
    }

    cudaDeviceSynchronize();

}
