#include <torch/torch.h>
#include <vector>
#include <unordered_set>
#include <algorithm>
#include <iostream>

std::unordered_set<int> create_edge_set(torch::Tensor& edges, std::vector<uint32_t>& edge_start, uint32_t vertex_index) {
    const auto vertex_count = edge_start.size();
    const auto edge_count = edges.size(0);

    const auto start_index = edge_start[vertex_index];
    const auto end_index = vertex_count == vertex_index + 1 ? edge_count : edge_start[vertex_index + 1];
    std::cout << "Start index: " << start_index << "    end index: " << end_index << "\n"; 

    std::unordered_set<int> set{};

    for (auto edge_index = start_index; edge_index < end_index; edge_index++) {
        std::cout << edge_index << "\n";
        set.insert(edges.accessor<long, 2>()[edge_index][1]);
    }

    std::cout << "Done with set\n";
    return set;
}

void graph_clustering(torch::Tensor& edges, torch::Tensor coeffs) {
    const auto edge_count = edges.size(0);
    const auto vertex_count = coeffs.size(0);

    // The row index where edges of the vertices start.
    std::vector<uint32_t> edge_starts(vertex_count, 0);
    std::vector<uint32_t> vertex_degree(vertex_count, 0);

    auto prev_vertex_index = -1;

    for (auto edge_index = 0; edge_index < edge_count; edge_index++) {
        std::cout << edge_index << "\n";
        auto vertex_index = edges.accessor<long, 2>()[edge_index][0];
        if (vertex_index != prev_vertex_index) {
            for (auto edge_starts_index = prev_vertex_index + 1; edge_starts_index <= vertex_index; edge_starts_index++) {
                edge_starts[edge_starts_index] = edge_index;
            }
            prev_vertex_index = vertex_index; 
        }
        vertex_degree[prev_vertex_index] += 1;
    }

    std::cout << "Found vertex degree\n";

    for (auto vertex_index = 0; vertex_index < edge_starts.size(); vertex_index++) {
        std::cout << "Creating edge set for vertex " << vertex_index << "\n";
        const auto edge_set = create_edge_set(edges, edge_starts, vertex_index);

        auto triangle_count = 0;

        for (auto neighbor : edge_set) {
            const auto neighbor_set = create_edge_set(edges, edge_starts, neighbor);
            const auto intersection_size = std::count_if(begin(edge_set), end(edge_set), [&](const auto& x) {
                return neighbor_set.find(x) != end(neighbor_set);
            });
            triangle_count += intersection_size;
        }
        const auto degree = vertex_degree[vertex_index];
        // 0 if t == 0 else t / (d * (d - 1)) for v, d, t, _ in td_iter
        coeffs.accessor<long, 1>()[vertex_index] = triangle_count == 0 ? 0 : triangle_count / (degree * (degree - 1));
    }
}