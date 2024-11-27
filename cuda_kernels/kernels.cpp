#include <iostream>
#include <torch/torch.h>

// adjacency.cu
void adjacency_matmul(
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, bool
);
void create_adjacency(torch::Tensor, torch::Tensor, torch::Tensor);

// feature_extraction.cu
void graph_features(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor);
void average_neighbor_features(
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor
);

// sinkhorn_log.cu
void sinkhorn_log_step(torch::Tensor, torch::Tensor, torch::Tensor, float);
float sinkhorn_log_marginal(torch::Tensor, torch::Tensor, torch::Tensor);

// distance.cu
void add_distance(torch::Tensor, torch::Tensor, torch::Tensor);

// regularize.cu
void regularize(torch::Tensor, torch::Tensor, int);

// update_quasi_permutation.cu
float update_quasi_permutation_log(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, float, float);

// hungarian.cu
void dense_hungarian(torch::Tensor, torch::Tensor);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // adjacency.cu
    m.def(
        "adjacency_matmul", &adjacency_matmul,
        "multiply adjacency matrix with tensor"
    );
    m.def(
        "create_adjacency", &create_adjacency, "create sparse adjacency matrix"
    );

    // feature_extraction.cu
    m.def("graph_features", &graph_features, "find features of graph");
    m.def(
        "average_neighbor_features", &average_neighbor_features,
        "find average of features among neighbors of each vertex in the graph"
    );

    // sinkhorn_log.cu
    m.def(
        "sinkhorn_log_step", &sinkhorn_log_step,
        "a single sinkhorn-knopp step in the log domain"
    );
    m.def(
        "sinkhorn_log_marginal", &sinkhorn_log_marginal,
        "calculate the marginal error in the log domain"
    );

    // distance.cu
    m.def("add_distance", &add_distance, "add euclidean distance to matrix");

    // regularize.cu
    m.def("regularize", &regularize, "regularize matrix");

    // update_quasi_permutation.cu
    m.def("update_quasi_permutation_log", &update_quasi_permutation_log, "update quasi permutation matrix");

    // hungarian.cu
    m.def("dense_hungarian", &dense_hungarian, "run dense hungarian");
}
