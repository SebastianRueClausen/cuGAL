#include "block.cuh"
#include <torch/torch.h>
#include <cuda.h>
#include <algorithm>

void data_stream_test(torch::Tensor matrix) {
    const long row_count = matrix.size(0);
    const long rows_per_block = 32;

    const long input_sizes[2] = { rows_per_block, matrix.size(1) };
    const long output_sizes[2] = { 1, 1 };
    auto blocks = create_max_possible_blocks(input_sizes, output_sizes, div_ceil(row_count, rows_per_block));

    long rows_uploaded = 0;
    for (long round = 0; rows_uploaded < row_count; round++) {
        auto block_index = round % blocks.size();
        auto row_amount = std::min(row_count - rows_uploaded, rows_per_block);
        blocks[block_index].upload(
            static_cast<float*>(matrix[rows_uploaded].data_ptr()),
            row_amount
        );
        rows_uploaded += row_amount;
    }

    cudaDeviceSynchronize();

    for (auto block : blocks) {
        block.destroy();
    }
}
