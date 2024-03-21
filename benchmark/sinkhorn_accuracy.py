from cugal import sinkhorn
from cugal.config import Config, SinkhornMethod
import matplotlib.pyplot as plt
import numpy as np
from fractions import Fraction
import math
import torch

def ground_truth_sinkhorn(K: np.ndarray) -> np.ndarray:
    na, _ = K.shape
    u = np.full(shape=(na,), fill_value=1/na)
    for _ in range(10000):
        v = 1 / (u @ K)
        u = 1 / (K @ v)
    return u.reshape(-1, 1) * K * v.reshape(1, -1)

def ground_truth(matrix: np.ndarray) -> np.ndarray:
    size = matrix.shape[0]
    matrix = matrix.astype('object')
    for i in range(size):
        for j in range(size):
            matrix[i, j] = math.exp(Fraction(matrix[i, j]) / -1)
    return ground_truth_sinkhorn(matrix)

def mean_difference(a, b: np.ndarray) -> float:
    return np.mean(abs(a - b))

def test(matrix: np.ndarray, config: Config) -> np.ndarray:
    return sinkhorn.sinkhorn(config.convert_tensor(torch.from_numpy(matrix)), config).numpy()

def test_accuracy():
    size = 128
    matrix = np.random.randn(size, size) * 20
    ground_truth_result = ground_truth(matrix)

    iteration_counts = [10, 20, 40, 80, 160, 320]

    double_results, log_results, mix_results = [], [], []

    for iteration_count in iteration_counts:
        double_results.append(mean_difference(ground_truth_result, test(matrix, Config(sinkhorn_threshold=1e-20, sinkhorn_iterations=iteration_count))))
        log_results.append(mean_difference(ground_truth_result, test(matrix, Config(sinkhorn_method=SinkhornMethod.LOG, dtype=torch.float32, sinkhorn_iterations=iteration_count, sinkhorn_threshold=1e-20))))
        mix_results.append(mean_difference(ground_truth_result, test(matrix, Config(sinkhorn_method=SinkhornMethod.MIX, dtype=torch.float32, sinkhorn_iterations=iteration_count, sinkhorn_threshold=1e-20))))

    plots = [
        (mix_results, 'mix float32'),
        (log_results, 'log float32'),
        (double_results, 'float64'),
    ]

    plt.figure()
    for results, label in plots:
        plt.plot(results, label=label)
    plt.title('accuracy')
    plt.xlabel('iteration count')
    plt.ylabel('mean difference from ground truth')
    plt.legend()
    plt.xticks(np.arange(len(iteration_counts)), iteration_counts)
    plt.show()


if __name__ == "__main__":
    test_accuracy()