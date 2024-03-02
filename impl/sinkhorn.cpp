#include <iostream>

void sinkhorn_step_cuda(torch::Tensor, torch::Tensor, torch::Tensor, int);

void sinkhorn_step(torch::Tensor K, torch::Tensor add, torch::Tensor out, int axis) {
    return sinkhorn_step_cuda(K, add, out, axis);
}