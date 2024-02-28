#include <iostream>

torch::Tensor sinkhorn_step_cuda(torch::Tensor, torch::Tensor, int);

torch::Tensor sinkhorn_step(torch::Tensor K, torch::Tensor add, int axis) {
    return sinkhorn_step_cuda(K, add, axis);
}