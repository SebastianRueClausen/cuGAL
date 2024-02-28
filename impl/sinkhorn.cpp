#include <iostream>

torch::Tensor logsumexp_cuda(torch::Tensor, int);

torch::Tensor logsumexp_row(torch::Tensor x) {
    return logsumexp_cuda(x, 0);
}

torch::Tensor logsumexp_col(torch::Tensor x) {
    return logsumexp_cuda(x, 1);
}