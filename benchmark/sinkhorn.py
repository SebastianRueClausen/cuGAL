import torch
import time
import numpy as np
import matplotlib.pyplot as plt
from cugal import sinkhorn
from functools import partial
from cugal.config import Config, SinkhornMethod


def mean_cuda_time(sinkhorn, iter_count: int) -> tuple[float, int]:
    times = []
    mean_required_iterations = 0

    for _ in range(iter_count):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        _, required_iterations = sinkhorn()
        mean_required_iterations += required_iterations
        end.record()

        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    return np.mean(times) / 1000, mean_required_iterations // iter_count


def mean_cpu_time(sinkhorn, iter_count: int) -> tuple[float, int]:
    times = []
    mean_required_iterations = 0

    for _ in range(iter_count):
        before = time.time()
        _, required_iterations = sinkhorn()
        mean_required_iterations += required_iterations
        times.append(time.time() - before)

    return np.mean(times), mean_required_iterations // iter_count


def benchmark_random_matrices():
    iter_count = 1
    matrix_sizes = [2500, 5000, 10000, 15000]

    gpu_config = Config(
        device="cuda", dtype=torch.float32, sinkhorn_iterations=200,
        sinkhorn_threshold=1e-7, sinkhorn_method=SinkhornMethod.LOG,
    )
    gpu_half_config = Config(
        device="cuda", dtype=torch.float16, sinkhorn_iterations=200,
        sinkhorn_threshold=1e-7, sinkhorn_method=SinkhornMethod.LOG,
    )
    cpu_config = Config(sinkhorn_iterations=200, sinkhorn_threshold=1e-7)

    cuda_results = []
    cuda_half_results = []
    torch_results = []
    cpu_results = []

    for matrix_size in matrix_sizes:
        matrix = torch.randn((matrix_size, matrix_size)) * 4.0

        cpu_matrix = cpu_config.convert_tensor(matrix)
        gpu_matrix = gpu_config.convert_tensor(matrix)
        gpu_half_matrix = gpu_half_config.convert_tensor(matrix)

        cpu_results.append(mean_cpu_time(
            partial(sinkhorn.sinkhorn_knopp, cpu_matrix, cpu_config),
            iter_count,
        ))

        torch_results.append(mean_cuda_time(
            partial(sinkhorn.sinkhorn_log, gpu_matrix, gpu_config),
            iter_count,
        ))

        cuda_results.append(mean_cuda_time(
            partial(sinkhorn.sinkhorn_log_cuda, gpu_matrix, gpu_config),
            iter_count,
        ))

        cuda_half_results.append(mean_cuda_time(
            partial(sinkhorn.sinkhorn_log_cuda, gpu_half_matrix, gpu_half_config),
            iter_count,
        ))

    plots = [
        (torch_results, 'log torch float32'),
        (cuda_results, 'log cuda float32'),
        (cuda_half_results, 'log cuda float16'),
        (cpu_results, 'cpu float64'),
    ]

    plt.figure()

    for results, label in plots:
        plt.plot([i for _, i in results], label=label)

    plt.title('iteration count')
    plt.xlabel('matrix size')
    plt.ylabel('iterations')

    plt.legend()

    plt.xticks(np.arange(len(matrix_sizes)), matrix_sizes)

    plt.figure()

    for results, label in plots:
        plt.plot([s for s, _ in results], label=label)

    plt.title('execution time')
    plt.xlabel('matrix size')
    plt.ylabel('seconds')

    plt.legend()

    plt.xticks(np.arange(len(matrix_sizes)), matrix_sizes)

    plt.show()


if __name__ == "__main__":
    benchmark_random_matrices()
