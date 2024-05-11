#include "CuLAP/include/culap.h"
#include "CuLAP/include/d_structs.h"
#include "CuLAP/include/d_vars.h"
#include "CuLAP/include/f_culap.h"
#include "CuLAP/include/f_cutils.h"
#include "CuLAP/include/f_culap.cu"
#include "CuLAP/include/f_cutils.cu"
#include "CuLAP/include/culap.cu"

#include <torch/torch.h>

void hungarian(torch::Tensor costs, torch::Tensor out)
{
    const auto size = costs.size(0);
    const auto batch_size = 1;
    const auto dev_id = 0;

    cudaSetDevice(dev_id);
    cudaDeviceSynchronize();

    CuLAP lpx(size, batch_size, dev_id);

    auto costs_ptr = reinterpret_cast<float *>(costs.data_ptr());
    lpx.solve(costs_ptr);

    auto out_ptr = reinterpret_cast<int *>(out.data_ptr());
    lpx.getAssignmentVector(out_ptr, 0);
}