#include <iostream>

torch::Tensor sinkhorn_step_cuda(torch::Tensor, torch::Tensor, int);

torch::Tensor sinkhorn_step_row(torch::Tensor K, torch::Tensor add) {
    return sinkhorn_step_cuda(K, add, 0);
}

torch::Tensor sinkhorn_step_col(torch::Tensor K, torch::Tensor add) {
    return sinkhorn_step_cuda(K, add, 1);
}