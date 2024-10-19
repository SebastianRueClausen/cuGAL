#include "../common.cuh"
#include <assert.h>
#include <cmath>
#include <cuda.h>
#include <random>
#include <set>
#include <stdio.h>
#include <thrust/detail/extrema.inl>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <time.h>
#include <utility>
#include <torch/torch.h>

struct Constants {
    uint32_t dimension, range, max_threads_per_block, block_count, data_block_size;

    Constants(uint32_t input_size) {
        dimension = input_size;
        max_threads_per_block = 1024;
        const auto columns_per_block = 512;
        block_count = div_ceil(dimension, columns_per_block);
        data_block_size = columns_per_block * dimension;
    }

    __device__ std::pair<int, int> linear_index_to_coordinates(int index) {
        return std::make_pair(index / dimension, index % dimension);
    }
};

// Data stored on the CPU side.
struct HostData {
    thrust::host_vector<int> column_of_star_at_row;
    thrust::host_vector<float> cost;

    HostData(Constants const &constants)
        : column_of_star_at_row(constants.dimension, 0),
          cost(constants.dimension * constants.dimension, 0) {}

    float get_cost(Constants const &constants, int row, int column) {
        return cost[row * constants.dimension + column];
    }

    void set_cost(Constants const &constants, int row, int column, float value) {
        cost[row * constants.dimension + column] = value;
    }
};

// Data stored on the GPU side.
struct DeviceData {
    DeviceData(Constants const &constants, float *slack)
        : slack(slack), min_in_rows(constants.dimension), min_in_cols(constants.dimension),
          zeros(constants.dimension * constants.dimension),
          zeros_in_block(constants.block_count, 0), row_of_star_at_column(constants.dimension, -1),
          column_of_star_at_row(constants.dimension, -1), cover_row(constants.dimension, 0),
          cover_column(constants.dimension, 0), column_of_prime_at_row(constants.dimension),
          row_of_green_at_column(constants.dimension) {}
    thrust::device_vector<float> min_in_rows, min_in_cols;
    thrust::device_vector<int> row_of_star_at_column, column_of_star_at_row,
        cover_row, cover_column, column_of_prime_at_row, row_of_green_at_column;
    thrust::device_vector<uint32_t> zeros, zeros_in_block;
    float *slack;
};

__managed__ __device__ uint32_t match_count;
__managed__ __device__ bool goto_5;
__managed__ __device__ bool repeat_kernel;

__global__ void calculate_min_in_rows(Constants constants, float *slack, float *min_in_rows) {
    const auto tid = threadIdx.x, bid = blockIdx.x;
    float min = INFINITY;
    for (auto column = tid; column < constants.dimension; column += blockDim.x) {
        min = std::min(min, slack[bid * constants.dimension + column]);
    }
    min = block_min_reduce(min);
    if (tid == 0)
        min_in_rows[bid] = min;
}

__global__ void calculate_min_in_columns(Constants constants, float *slack, float *min_in_cols) {
    const auto tid = threadIdx.x, bid = blockIdx.x;
    float min = INFINITY;
    for (auto row = tid; row < constants.dimension; row += blockDim.x) {
        min = std::min(min, slack[row * constants.dimension + bid]);
    }
    min = block_min_reduce(min);
    if (tid == 0)
        min_in_cols[bid] = min;
}

__global__ void step_1_row_sub(Constants constants, float *slack, float *min_in_rows) {
    const auto index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= constants.dimension * constants.dimension)
        return;
    slack[index] -= min_in_rows[index / constants.dimension];
}

__global__ void step_1_col_sub(Constants constants, float *slack, float *min_in_cols) {
    const auto index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= constants.dimension * constants.dimension)
        return;
    slack[index] -= min_in_cols[index % constants.dimension];
}

__global__ void compress_matrix(Constants constants, float *slack, uint32_t *zeros, uint32_t *zeros_in_block) {
    const auto index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= constants.dimension * constants.dimension)
        return;
    if (std::abs(slack[index]) < 0.0001) {
        const auto block = index / constants.data_block_size;
        const auto base_index = block * constants.data_block_size;
        const auto offset = atomicAdd(zeros_in_block + block, 1);
        zeros[base_index + offset] = index;
    }
}

__global__ void step_2(
    Constants constants, uint32_t *zeros, uint32_t *zeros_in_block, int *row_of_star_at_column,
    int *column_of_star_at_row, int *cover_row, int *cover_column
) {
    const auto tid = threadIdx.x, block_index = blockIdx.x;
    __shared__ bool repeat, s_repeat_kernel;
    if (tid == 0)
        s_repeat_kernel = false;
    do {
        __syncthreads();
        if (tid == 0)
            repeat = false;
        __syncthreads();
        for (auto block_offset = tid; block_offset < zeros_in_block[block_index];
             block_offset += blockDim.x) {
            const auto zeros_index = (block_index * constants.data_block_size) + block_offset;
            if (zeros_index >= constants.dimension * constants.dimension)
                continue;
            const auto [row, column] = constants.linear_index_to_coordinates(zeros[zeros_index]);
            if (cover_row[row] == 0 && cover_column[column] == 0) {
                // Try to take row.
                if (!atomicExch((int *)&(cover_row[row]), 1)) {
                    // Try to take column.
                    if (!atomicExch((int *)&(cover_column[column]), 1)) {
                        row_of_star_at_column[column] = row;
                        column_of_star_at_row[row] = column;
                    } else {
                        // If another thread took the column, give up the row and try over.
                        cover_row[row] = 0;
                        repeat = true;
                        s_repeat_kernel = true;
                    }
                }
            }
        }
        __syncthreads();
    } while (repeat);
    if (s_repeat_kernel)
        repeat_kernel = true;
}

__global__ void step_3(Constants constants, int *row_of_star_at_column, int *cover_column) {
    const auto index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= constants.dimension)
        return;
    if (row_of_star_at_column[index] >= 0) {
        cover_column[index] = 1;
        atomicAdd((int *)&match_count, 1);
    }
}

__global__ void step_4(
    Constants constants, uint32_t *zeros, uint32_t *zeros_in_block, int *column_of_star_at_row,
    int *cover_row, int *cover_column, int *column_of_prime_at_row
) {
    __shared__ bool s_found, s_goto_5, s_repeat_kernel;
    volatile int *v_cover_row = cover_row, *v_cover_column = cover_column;
    const auto index = threadIdx.x, block_index = blockIdx.x;
    if (index == 0) {
        s_repeat_kernel = false;
        s_goto_5 = false;
    }
    do {
        __syncthreads();
        if (index == 0)
            s_found = false;
        __syncthreads();
        for (auto block_offset = index; block_offset < zeros_in_block[block_index];
             block_offset += blockDim.x) {
            const auto zeros_index = (block_index * constants.data_block_size) + block_offset;
            if (zeros_index >= constants.dimension * constants.dimension)
                continue;
            const auto [row, column] = constants.linear_index_to_coordinates(zeros[zeros_index]);
            const auto starred_column = column_of_star_at_row[row];
            for (auto n = 0; n < 10; n++) {
                if (!v_cover_column[column] && !v_cover_row[row]) {
                    s_found = true;
                    s_repeat_kernel = true;
                    column_of_prime_at_row[row] = column;
                    if (starred_column >= 0) {
                        v_cover_row[row] = 1;
                        __threadfence();
                        v_cover_column[starred_column] = 0;
                    } else {
                        s_goto_5 = true;
                    }
                }
            }
        }
        __syncthreads();
    } while (s_found && !s_goto_5);
    if (index == 0 && s_repeat_kernel)
        repeat_kernel = true;
    if (index == 0 && s_goto_5)
        goto_5 = true;
}

__global__ void step_5a(
    Constants constants, int *row_of_star_at_column, int *column_of_star_at_row,
    int *column_of_prime_at_row, int *row_of_green_at_column
) {
    const auto index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= constants.dimension)
        return;
    int row_z0, column_z0;
    column_z0 = column_of_prime_at_row[index];
    if (column_z0 >= 0 && column_of_star_at_row[index] < 0) {
        row_of_green_at_column[column_z0] = index;
        while ((row_z0 = row_of_star_at_column[column_z0]) >= 0) {
            column_z0 = column_of_prime_at_row[row_z0];
            row_of_green_at_column[column_z0] = row_z0;
        }
    }
}

__global__ void step_5b(
    Constants constants, int *row_of_star_at_column, int *column_of_star_at_row,
    int *row_of_green_at_column
) {
    const auto index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= constants.dimension)
        return;
    int row_z0, column_z0, column_z2;
    row_z0 = row_of_green_at_column[index];
    if (row_z0 >= 0 && row_of_star_at_column[index] < 0) {
        column_z2 = column_of_star_at_row[row_z0];
        column_of_star_at_row[row_z0] = index;
        row_of_star_at_column[index] = row_z0;
        while (column_z2 >= 0) {
            row_z0 = row_of_green_at_column[column_z2]; // row of Z2
            column_z0 = column_z2;                      // col of Z2
            column_z2 = column_of_star_at_row[row_z0];  // col of Z4
            column_of_star_at_row[row_z0] = column_z0;
            row_of_star_at_column[column_z0] = row_z0;
        }
    }
}

__global__ void min_in_uncovered_rows(
    Constants constants, float *slack, float *min_in_rows, uint32_t *row_indices, int *cover_column
) {
    const auto tid = threadIdx.x, bid = blockIdx.x;
    const auto row = row_indices[bid];
    float min = INFINITY;
    for (auto column = tid; column < constants.dimension; column += blockDim.x) {
        min = std::min(
            min, cover_column[column] == 1 ? INFINITY : slack[row * constants.dimension + column]
        );
    }
    min = block_min_reduce(min);
    if (tid == 0)
        min_in_rows[bid] = min;
}

__global__ void step_6_add_sub(
    Constants constants, float *slack, int *cover_row, int *cover_column, float *matrix_minimum
) {
    const auto column = blockDim.x * blockIdx.x + threadIdx.x;
    const auto row = blockDim.y * blockIdx.y + threadIdx.y;
    if (column >= constants.dimension || row >= constants.dimension)
        return;
    const auto index = row * constants.dimension + column;
    if (cover_row[row] == 1 && cover_column[column] == 1)
        slack[index] += *matrix_minimum;
    if (cover_row[row] == 0 && cover_column[column] == 0)
        slack[index] -= *matrix_minimum;
}

inline cudaError_t check_cuda(cudaError_t result) {
    if (result != cudaSuccess) {
        printf("CUDA Runtime Error: %s\n", cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
    return result;
};

constexpr uint32_t default_thread_count = 128;

void hungarian(Constants constants, DeviceData &device_data) {
    calculate_min_in_rows<<<constants.dimension, default_thread_count>>>(
        constants, device_data.slack, device_data.min_in_rows.data().get()
    );
    check_cuda(cudaDeviceSynchronize());
    step_1_row_sub<<<
        div_ceil(constants.dimension * constants.dimension, default_thread_count),
        default_thread_count>>>(
        constants, device_data.slack, device_data.min_in_rows.data().get()
    );
    check_cuda(cudaDeviceSynchronize());
    calculate_min_in_columns<<<constants.dimension, default_thread_count>>>(
        constants, device_data.slack, device_data.min_in_cols.data().get()
    );
    check_cuda(cudaDeviceSynchronize());
    step_1_col_sub<<<
        div_ceil(constants.dimension * constants.dimension, default_thread_count),
        default_thread_count>>>(
        constants, device_data.slack, device_data.min_in_cols.data().get()
    );
    check_cuda(cudaDeviceSynchronize());

    compress_matrix<<<
        div_ceil(constants.dimension * constants.dimension, default_thread_count),
        default_thread_count>>>(
        constants, device_data.slack, device_data.zeros.data().get(),
        device_data.zeros_in_block.data().get()
    );
    check_cuda(cudaDeviceSynchronize());

    do {
        repeat_kernel = false;
        step_2<<<constants.block_count, constants.max_threads_per_block>>>(
            constants, thrust::raw_pointer_cast(device_data.zeros.data()),
            device_data.zeros_in_block.data().get(), device_data.row_of_star_at_column.data().get(),
            device_data.column_of_star_at_row.data().get(), device_data.cover_row.data().get(),
            device_data.cover_column.data().get()
        );
        check_cuda(cudaDeviceSynchronize());
    } while (repeat_kernel);

    while (1) {
        thrust::fill(thrust::device, device_data.cover_row.begin(), device_data.cover_row.end(), 0);
        thrust::fill(
            thrust::device, device_data.cover_column.begin(), device_data.cover_column.end(), 0
        );
        match_count = 0;
        step_3<<<div_ceil(constants.dimension, default_thread_count), default_thread_count>>>(
            constants, device_data.row_of_star_at_column.data().get(),
            device_data.cover_column.data().get()
        );
        check_cuda(cudaDeviceSynchronize());

        if (match_count >= constants.dimension)
            break;

        thrust::fill(
            thrust::device, device_data.column_of_prime_at_row.begin(),
            device_data.column_of_prime_at_row.end(), -1
        );
        thrust::fill(
            thrust::device, device_data.row_of_green_at_column.begin(),
            device_data.row_of_green_at_column.end(), -1
        );

        while (1) {
            do {
                goto_5 = false;
                repeat_kernel = false;
                step_4<<<constants.block_count, constants.max_threads_per_block>>>(
                    constants, device_data.zeros.data().get(),
                    device_data.zeros_in_block.data().get(),
                    device_data.column_of_star_at_row.data().get(),
                    device_data.cover_row.data().get(), device_data.cover_column.data().get(),
                    device_data.column_of_prime_at_row.data().get()
                );
                check_cuda(cudaDeviceSynchronize());
            } while (repeat_kernel && !goto_5);

            if (goto_5)
                break;

            thrust::device_vector<uint32_t> uncovered_rows(constants.dimension);
            const auto end = thrust::copy_if(
                thrust::device, thrust::make_counting_iterator(0u),
                thrust::make_counting_iterator(constants.dimension), device_data.cover_row.begin(),
                uncovered_rows.begin(), thrust::placeholders::_1 == 0
            );
            uncovered_rows.resize(thrust::distance(uncovered_rows.begin(), end));

            min_in_uncovered_rows<<<uncovered_rows.size(), default_thread_count>>>(
                constants, device_data.slack, device_data.min_in_rows.data().get(),
                uncovered_rows.data().get(), device_data.cover_column.data().get()
            );
            auto min_element = thrust::min_element(
                thrust::device, device_data.min_in_rows.begin(),
                device_data.min_in_rows.begin() + uncovered_rows.size()
            );

            const auto blocks =
                dim3(div_ceil(constants.dimension, 32), div_ceil(constants.dimension, 32), 1);
            const auto threads = dim3(32, 32, 1);
            step_6_add_sub<<<blocks, threads>>>(
                constants, device_data.slack, device_data.cover_row.data().get(),
                device_data.cover_column.data().get(), min_element.base().get()
            );
            check_cuda(cudaDeviceSynchronize());

            thrust::fill(thrust::device, device_data.zeros.begin(), device_data.zeros.end(), 0);
            thrust::fill(
                thrust::device, device_data.zeros_in_block.begin(),
                device_data.zeros_in_block.end(), 0
            );

            compress_matrix<<<
                div_ceil(constants.dimension * constants.dimension, default_thread_count),
                default_thread_count>>>(
                constants, device_data.slack, device_data.zeros.data().get(),
                device_data.zeros_in_block.data().get()
            );
            check_cuda(cudaDeviceSynchronize());
        }

        step_5a<<<div_ceil(constants.dimension, default_thread_count), default_thread_count>>>(
            constants, device_data.row_of_star_at_column.data().get(),
            device_data.column_of_star_at_row.data().get(),
            device_data.column_of_prime_at_row.data().get(),
            device_data.row_of_green_at_column.data().get()
        );
        check_cuda(cudaDeviceSynchronize());

        step_5b<<<div_ceil(constants.dimension, default_thread_count), default_thread_count>>>(
            constants, device_data.row_of_star_at_column.data().get(),
            device_data.column_of_star_at_row.data().get(),
            device_data.row_of_green_at_column.data().get()
        );
        check_cuda(cudaDeviceSynchronize());
    }
}

void dense_hungarian(torch::Tensor cost, torch::Tensor output) {
    Constants constants(cost.size(0));
    DeviceData device_data(constants, cost.data_ptr<float>());
    hungarian(constants, device_data);
    auto dst = output.data_ptr<int>();
    check_cuda(cudaMemcpy(
        dst, device_data.column_of_star_at_row.data().get(),
        constants.dimension * sizeof(float), cudaMemcpyDeviceToHost)
    );
}

/*
int main() {
    Constants constants(1000);

    HostData host_data(constants);
    DeviceData device_data(constants);

    std::default_random_engine generator(1337);
    std::uniform_real_distribution<float> distribution(0.0, 1.0);

    for (auto row = 0; row < constants.dimension; row++) {
        for (auto column = 0; column < constants.dimension; column++) {
            host_data.set_cost(constants, row, column, distribution(generator));
        }
    }

    device_data.slack = host_data.cost;

    time_t start_time = clock();
    hungarian(constants, device_data);
    check_cuda(cudaDeviceSynchronize());
    time_t stop_time = clock();

    host_data.column_of_star_at_row = device_data.column_of_star_at_row;

    // Check that all the columns have been assigned.
    std::set<int> column_set(
        host_data.column_of_star_at_row.begin(), host_data.column_of_star_at_row.end()
    );
    assert(column_set.size() == constants.dimension);

    auto total_cost = 0.0;
    for (auto row = 0; row < constants.dimension; row++) {
        const auto column = host_data.column_of_star_at_row[row];
        assert(column >= 0);
        assert(column < constants.dimension);
        total_cost += host_data.get_cost(constants, row, column);
    }

    printf("Total cost is \t %f\n", total_cost);
    printf(
        "Low resolution time is \t %f \n",
        1000.0 * (double)(stop_time - start_time) / CLOCKS_PER_SEC
    );
}
*/
