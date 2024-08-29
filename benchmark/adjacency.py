"""Benchmark memory usage and computation time of using sparse versus dense adjacency matrices."""

from cugal.adjacency import Adjacency
from cugal.profile import TimeStamp
import networkx as nx
import torch
import matplotlib.pyplot as plt
import numpy as np


def random_test_data(node_count: int) -> tuple[torch.Tensor, torch.Tensor]:
    matrix = torch.randn(size=(node_count, node_count), dtype=torch.float32)
    A = torch.from_numpy(nx.to_numpy_array(
        nx.newman_watts_strogatz_graph(node_count, node_count // 16, 0.01), dtype=np.float32
    ))
    return A, matrix


def time(function, device: str) -> tuple[float, any]:
    start = TimeStamp(device)
    function()
    return TimeStamp(device).elapsed_seconds(start)


def benchmark_random_graphs(graph_sizes: list[int]):
    cuda_times, dense_torch_times, dense_cpu_times = [], [], []
    sparse_sizes, dense_sizes = [], []

    for node_count in graph_sizes:
        adjacency, matrix = random_test_data(node_count)
        dense_cpu_times.append(time(lambda: adjacency @ matrix, "cpu"))

        adjacency = adjacency.to(device="cuda")
        matrix = matrix.to(device="cuda")

        dense_torch_times.append(time(lambda: adjacency @ matrix, "cpu"))

        adjacency = Adjacency.from_dense(adjacency)
        cuda_times.append(time(lambda: adjacency.mul(matrix), "cuda"))

        dense_sizes.append(node_count * node_count * 4)
        sparse_sizes.append(adjacency.byte_size())

    plots = [
        (cuda_times, 'sparse cuda'),
        (dense_torch_times, 'dense torch'),
        (dense_cpu_times, 'dense cpu'),
    ]

    plt.figure()
    for results, label in plots:
        plt.plot(results, label=label)
    plt.title('execution time')
    plt.xlabel('graph size')
    plt.ylabel('seconds')
    plt.legend()
    plt.xticks(np.arange(len(graph_sizes)), graph_sizes)

    plt.figure()
    plt.plot(dense_sizes, label="dense")
    plt.plot(sparse_sizes, label="sparse")
    plt.title('memory usage')
    plt.xlabel('graph size')
    plt.ylabel('bytes')
    plt.legend()
    plt.xticks(np.arange(len(graph_sizes)), graph_sizes)

    plt.show()


if __name__ == "__main__":
    benchmark_random_graphs([2500, 5000, 10000, 15000])
