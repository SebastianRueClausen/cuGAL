from cugal.adjacency import Adjacency
import networkx as nx
from util import cpu_time, cuda_time
import torch
import matplotlib.pyplot as plt
import numpy as np


def random_test_data(node_count: int) -> tuple[torch.Tensor, torch.Tensor]:
    matrix = torch.randn(size=(node_count, node_count), dtype=torch.float32)
    A = torch.from_numpy(nx.to_numpy_array(
        nx.newman_watts_strogatz_graph(node_count, 7, 0.01), dtype=np.float32
    ))
    return A, matrix


def benchmark_random_graphs(graph_sizes: list[int]):
    cuda_times, dense_torch_times, dense_cpu_times = [], [], []
    sparse_sizes, dense_sizes = [], []

    for node_count in graph_sizes:
        adjacency, matrix = random_test_data(node_count)
        dense_cpu_times.append(cpu_time(lambda: adjacency @ matrix)[0])

        adjacency = adjacency.to(device="cuda")
        matrix = matrix.to(device="cuda")

        dense_torch_times.append(cuda_time(lambda: adjacency @ matrix)[0])

        adjacency = Adjacency.from_dense(adjacency)
        cuda_times.append(cuda_time(lambda: adjacency.mul(matrix))[0])

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
