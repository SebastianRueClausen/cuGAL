#include <iostream>
#include <torch/torch.h>

void sinkhorn_step_cuda(torch::Tensor, torch::Tensor, torch::Tensor);
void adjacency_matmul_cuda(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, bool);
void create_adjacency_cuda(torch::Tensor, torch::Tensor, torch::Tensor);
void graph_features(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor);
void average_neighbor_features(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor);

void adjacency_matmul(
  torch::Tensor col_indices,
  torch::Tensor row_pointers,
  torch::Tensor matrix,
  torch::Tensor out,
  bool negate_lhs
) {
  adjacency_matmul_cuda(col_indices, row_pointers, matrix, out, negate_lhs);
}

void create_adjacency(torch::Tensor edges, torch::Tensor col_indices, torch::Tensor row_pointers) {
  create_adjacency_cuda(edges, col_indices, row_pointers);
}

void sinkhorn_step(torch::Tensor K, torch::Tensor add, torch::Tensor out) {
  sinkhorn_step_cuda(K, add, out);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sinkhorn_step", &sinkhorn_step, "sinkhorn step");
  m.def("adjacency_matmul", &adjacency_matmul, "multiply adjacency matrix with tensor");
  m.def("create_adjacency", &create_adjacency, "create sparse adjacency matrix");
  m.def("graph_features", &graph_features, "find features of graph");
  m.def("average_neighbor_features", &average_neighbor_features, "find average of features among neighbors of each vertex in the graph");
}
