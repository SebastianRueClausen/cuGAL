#include <iostream>
#include <torch/torch.h>

void sinkhorn_step_cuda(torch::Tensor, torch::Tensor, torch::Tensor);
void adjacency_matmul_cuda(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, bool);

void adjacency_matmul(
  torch::Tensor col_indices,
  torch::Tensor row_pointers,
  torch::Tensor matrix,
  torch::Tensor out,
  bool negate_lhs
) {
  return adjacency_matmul_cuda(col_indices, row_pointers, matrix, out, negate_lhs);
}

void sinkhorn_step(torch::Tensor K, torch::Tensor add, torch::Tensor out) {
  return sinkhorn_step_cuda(K, add, out);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sinkhorn_step", &sinkhorn_step, "sinkhorn step");
  m.def("adjacency_matmul", &adjacency_matmul, "multiply adjacency matrix with tensor");
}