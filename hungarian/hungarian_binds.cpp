#include <iostream>
#include <torch/torch.h>
#include <vector>

void hungarian(float *cost, int size, int *out);

void hungarian_torch(torch::Tensor cost, torch::Tensor out)
{
    hungarian(cost.data_ptr<float>(), cost.size(0), out.data_ptr<int>());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("hungarian_torch", &hungarian_torch, "hungarian algorithm");
}