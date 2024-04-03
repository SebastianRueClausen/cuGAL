import torch
import numpy as np
import matplotlib.pyplot as plt
from cugal import sinkhorn
from cugal.config import Config, SinkhornMethod
from cugal.profile import SinkhornProfile

def benchmark_random_matrices(matrix_sizes: list[int]):
    gpu_config = Config(
        device="cuda", dtype=torch.float32, sinkhorn_iterations=200,
        sinkhorn_threshold=1e-7, sinkhorn_method=SinkhornMethod.LOG,
    )
    gpu_half_config = Config(
        device="cuda", dtype=torch.float16, sinkhorn_iterations=200,
        sinkhorn_threshold=1e-7, sinkhorn_method=SinkhornMethod.LOG,
    )
    cpu_config = Config(sinkhorn_iterations=200, sinkhorn_threshold=1e-7)

    log_profiles, log_half_profiles, mix_profiles, cpu_profiles = [], [], [], []

    for matrix_size in matrix_sizes:
        matrix = torch.randn((matrix_size, matrix_size)) * 4.0

        profile = SinkhornProfile()
        sinkhorn.sinkhorn_knopp(cpu_config.convert_tensor(matrix), cpu_config, profile)
        log_profiles.append(profile)

        gpu_matrix = gpu_config.convert_tensor(matrix)

        profile = SinkhornProfile()
        sinkhorn.mixhorn(gpu_matrix, gpu_config, profile)
        mix_profiles.append(profile)

        profile = SinkhornProfile()
        sinkhorn.loghorn(gpu_matrix, gpu_config, profile)
        log_profiles.append(profile)

        profile = SinkhornProfile()
        sinkhorn.loghorn(gpu_half_config.convert_tensor(matrix), gpu_half_config, profile)
        log_half_profiles.append(profile)

    plots = [
        (mix_profiles, 'mix float32'),
        (log_profiles, 'log float32'),
        (log_half_profiles, 'log float16'),
        (cpu_profiles, 'cpu float64'),
    ]

    plt.figure()
    for profiles, label in plots:
        plt.plot([profile.iteration_count for profile in profiles], label=label)
    plt.title('iteration count')
    plt.xlabel('matrix size')
    plt.ylabel('iterations')
    plt.legend()
    plt.xticks(np.arange(len(matrix_sizes)), matrix_sizes)

    plt.figure()
    for profiles, label in plots:
        plt.plot([profile.time for profile, _ in profiles], label=label)
    plt.title('execution time')
    plt.xlabel('matrix size')
    plt.ylabel('seconds')
    plt.legend()
    plt.xticks(np.arange(len(matrix_sizes)), matrix_sizes)

    plt.show()


if __name__ == "__main__":
    benchmark_random_matrices([2500, 5000, 10000, 15000])
