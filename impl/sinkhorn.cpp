#include <iostream>

void sinkhorn_step_cuda(torch::Tensor, torch::Tensor, torch::Tensor);

void sinkhorn_step(torch::Tensor K, torch::Tensor add, torch::Tensor out) {
    return sinkhorn_step_cuda(K, add, out);
}