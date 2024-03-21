import torch
import numpy as np
import matplotlib.pyplot as plt
from cugal import sinkhorn
from functools import partial
from cugal.config import Config, SinkhornMethod
from util import cpu_time, cuda_time


def mean_cuda_time(sinkhorn, iter_count: int) -> tuple[float, int]:
    times = []
    mean_required_iterations = 0

    for _ in range(iter_count):
        elapsed, (_, required_iterations) = cuda_time(sinkhorn)
        times.append(elapsed)
        mean_required_iterations += required_iterations

    return np.mean(times), mean_required_iterations // iter_count


def mean_cpu_time(sinkhorn, iter_count: int) -> tuple[float, int]:
    times = []
    mean_required_iterations = 0

    for _ in range(iter_count):
        elapsed, (_, required_iterations) = cpu_time(sinkhorn)
        mean_required_iterations += required_iterations
        times.append(elapsed)

    return np.mean(times), mean_required_iterations // iter_count


def benchmark_random_matrices(matrix_sizes: list[int], iter_count: int):
    gpu_config = Config(
        device="cuda", dtype=torch.float32, sinkhorn_iterations=200,
        sinkhorn_threshold=1e-7, sinkhorn_method=SinkhornMethod.LOG,
    )
    gpu_half_config = Config(
        device="cuda", dtype=torch.float16, sinkhorn_iterations=200,
        sinkhorn_threshold=1e-7, sinkhorn_method=SinkhornMethod.LOG,
    )
    cpu_config = Config(sinkhorn_iterations=200, sinkhorn_threshold=1e-7)

    log_results, log_half_results, mix_results, cpu_results = [], [], [], [], []

    for matrix_size in matrix_sizes:
        matrix = torch.randn((matrix_size, matrix_size)) * 4.0

        cpu_matrix = cpu_config.convert_tensor(matrix)
        gpu_matrix = gpu_config.convert_tensor(matrix)
        gpu_half_matrix = gpu_half_config.convert_tensor(matrix)

        cpu_results.append(mean_cpu_time(
            partial(sinkhorn.sinkhorn_knopp, cpu_matrix, cpu_config),
            iter_count,
        ))

        mix_results.append(mean_cuda_time(
            partial(sinkhorn.mixhorn, gpu_matrix, gpu_config),
            iter_count,
        ))

        log_results.append(mean_cuda_time(
            partial(sinkhorn.loghorn, gpu_matrix, gpu_config),
            iter_count,
        ))

        log_half_results.append(mean_cuda_time(
            partial(sinkhorn.sinkhorn_log_cuda,
                    gpu_half_matrix, gpu_half_config),
            iter_count,
        ))

    plots = [
        (mix_results, 'mix float32'),
        (log_results, 'log float32'),
        (log_half_results, 'log float16'),
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
    benchmark_random_matrices([2500, 5000, 10000, 15000], 1)
