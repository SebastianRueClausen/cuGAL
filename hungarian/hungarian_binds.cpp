#include <iostream>
#include <torch/torch.h>
#include <vector>

std::vector<int> hungarian(const std::vector<std::vector<float>> &);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("hungarian", &hungarian, "hungarian algorithm");
}