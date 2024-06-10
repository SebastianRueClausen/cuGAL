#include <iostream>
#include <torch/torch.h>

void sinkhorn_step_cuda(torch::Tensor, torch::Tensor, torch::Tensor, bool);
void adjacency_matmul(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, bool);
void create_adjacency_cuda(torch::Tensor, torch::Tensor, torch::Tensor);
void graph_features(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor);
void average_neighbor_features(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor);
void add_distance(torch::Tensor, torch::Tensor, torch::Tensor);
void regularize(torch::Tensor, torch::Tensor, int);
void sinkhorn_step_cols(torch::Tensor, torch::Tensor, torch::Tensor);

void create_adjacency(torch::Tensor edges, torch::Tensor col_indices, torch::Tensor row_pointers)
{
  create_adjacency_cuda(edges, col_indices, row_pointers);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("sinkhorn_step", &sinkhorn_step_cuda, "sinkhorn step");
  m.def("sinkhorn_step_cols", &sinkhorn_step_cols, "sinkhorn step cols");
  m.def("adjacency_matmul", &adjacency_matmul, "multiply adjacency matrix with tensor");
  m.def("create_adjacency", &create_adjacency, "create sparse adjacency matrix");
  m.def("graph_features", &graph_features, "find features of graph");
  m.def("average_neighbor_features", &average_neighbor_features, "find average of features among neighbors of each vertex in the graph");
  m.def("add_distance", &add_distance, "add euclidean distance to matrix");
  m.def("regularize", &regularize, "regularize matrix");
}
