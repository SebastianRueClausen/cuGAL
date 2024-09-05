import torch
import matplotlib.pyplot as plt


def heatmap(matrix: torch.Tensor):
    plt.imshow(matrix.cpu().numpy(), cmap='hot', interpolation='nearest')
    plt.show()


def sparsity(matrix: torch.Tensor) -> float:
    return torch.sum(abs(matrix) < 0.001).item() / (matrix.shape[0] * matrix.shape[1])
