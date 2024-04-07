#include <torch/torch.h>
#include <vector>
#include <unordered_set>
#include <algorithm>
#include <iostream>
#include <utility>

std::pair<int, int> edge_range(torch::Tensor& edges, std::vector<uint32_t>& edge_starts, uint32_t vertex_index) {
    return std::make_pair(
        edge_starts[vertex_index],
        edge_starts.size() == vertex_index + 1 ? edges.size(0) : edge_starts[vertex_index + 1]
    );
}

long intersection_size(torch::Tensor& edges, std::vector<uint32_t>& edge_starts, uint32_t vertex_a, uint32_t vertex_b) {
    const auto [a_start, a_end] = edge_range(edges, edge_starts, vertex_a);
    const auto [b_start, b_end] = edge_range(edges, edge_starts, vertex_b);

    long intersection_size = 0;

    for (auto a_index = a_start, b_index = b_start; a_index < a_end && b_index < b_end;) {
        const auto a_edge = edges.accessor<long, 2>()[a_index][1];
        const auto b_edge = edges.accessor<long, 2>()[b_index][1];

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

void graph_clustering(torch::Tensor& edges, torch::Tensor coeffs) {
    const auto edge_count = edges.size(0);
    const auto vertex_count = coeffs.size(0);

    // The row index where edges of the vertices start.
    std::vector<uint32_t> edge_starts(vertex_count, 0);
    std::vector<uint32_t> vertex_degree(vertex_count, 0);

    auto prev_vertex_index = -1;

    for (auto edge_index = 0; edge_index < edge_count; edge_index++) {
        auto vertex_index = edges.accessor<long, 2>()[edge_index][0];
        if (vertex_index != prev_vertex_index) {
            for (auto edge_starts_index = prev_vertex_index + 1; edge_starts_index <= vertex_index; edge_starts_index++) {
                edge_starts[edge_starts_index] = edge_index;
            }
            prev_vertex_index = vertex_index;
        }
        vertex_degree[prev_vertex_index] += 1;
    }

    for (auto vertex_index = 0; vertex_index < edge_starts.size(); vertex_index++) {
        auto triangle_count = 0;

        const auto [start, end] = edge_range(edges, edge_starts, vertex_index);
        for (auto edge_index = start; edge_index < end; edge_index++) {
            const auto neighbor_index = edges.accessor<long, 2>()[edge_index][1];
            triangle_count += intersection_size(edges, edge_starts, vertex_index, neighbor_index);
        }

        const auto degree = vertex_degree[vertex_index];

        // 0 if t == 0 else t / (d * (d - 1)) for v, d, t, _ in td_iter
        const auto result = triangle_count == 0 ? 0.0 : float(triangle_count) / (degree * (degree - 1));
        coeffs.accessor<float, 1>()[vertex_index] = result;
    }
}
