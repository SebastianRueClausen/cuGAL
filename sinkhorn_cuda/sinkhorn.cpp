#include <iostream>
#include <torch/torch.h>

void sinkhorn_step_cuda(torch::Tensor, torch::Tensor, torch::Tensor);

void sinkhorn_step(torch::Tensor K, torch::Tensor add, torch::Tensor out) {
    return sinkhorn_step_cuda(K, add, out);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sinkhorn_step", &sinkhorn_step, "sinkhorn step");
}