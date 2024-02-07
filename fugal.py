from dataclasses import dataclass
import numpy as np
from hungarian import hungarian_algorithm
from sinkhorn_knopp import sinkhorn_knopp as skp

def create_random_adjacency_matrix(
    vertex_count: int,
    generator: np.random.Generator,
    connectivity: float = 0.4,
) -> np.ndarray:
    graph = np.zeros(shape=(vertex_count, vertex_count), dtype=np.float32)
    for a in range(vertex_count):
        for b in range(0, a):
            value = generator.choice([1, 0], p=[connectivity, 1.0 - connectivity])
            graph[a, b] = value
            graph[b, a] = value
    return graph

def permute_adjacency_matrix(
    graph: np.ndarray,
    generator: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    vertex_count, _ = graph.shape
    permuted = np.zeros(shape=graph.shape)

    permutation = np.arange(vertex_count)
    generator.shuffle(permutation)

    for i in range(vertex_count):
        for j in range(vertex_count):
            permuted[i, j] = graph[permutation[i], permutation[j]]

    return permuted, permutation

def extract_features(adjacency: np.ndarray) -> np.ndarray:
    vertex_count, _ = adjacency.shape

    vertex_1st_degrees = adjacency.sum(axis=0)
    vertex_2nd_degrees = []

    for row in adjacency:
        neighbours = np.nonzero(row)[0]
        neighbour_connection_count = sum(
            np.dot(adjacency[neighbour], row) for neighbour in neighbours
        )
        vertex_2nd_degrees.append(neighbour_connection_count)

    denom = vertex_1st_degrees * (vertex_1st_degrees - 1)
    vertex_clustering = np.divide(vertex_2nd_degrees, denom, where=denom != 0)

    mean_neighbour_degress = (vertex_1st_degrees @ adjacency) / vertex_count
    mean_neighbour_clustering = (vertex_clustering @ adjacency) / vertex_count

    return np.vstack([
        vertex_1st_degrees,
        vertex_clustering,
        mean_neighbour_degress,
        mean_neighbour_clustering,
    ]).T

def euclidian_distance(f1: np.ndarray, f2: np.ndarray) -> np.ndarray:
    p1 = np.sum(f1**2, axis=1)[:, np.newaxis]
    p2 = np.sum(f2**2, axis=1)
    p3 = -2 * np.dot(f1, f2.T)
    return np.sqrt(p1 + p2 + p3)

def find_quasi_permutation(
    g1: np.ndarray,
    g2: np.ndarray,
    distance: np.ndarray,
    my: float,
    iteration_count: int,
) -> np.ndarray:
    vertex_count, _ = g1.shape
    q = np.ones(vertex_count) - np.ones(vertex_count) / vertex_count

    f_deriv = lambda x: g1 @ x @ g2.T + my * distance.T
    g_deriv = lambda x: np.ones(shape=g1.shape) - x

    for iteration in range(iteration_count):
        for inner in range(10):
            grad = f_deriv(q) + iteration * g_deriv(q)
            optimizer = skp.SinkhornKnopp()
            out = optimizer.fit(grad)
            alpha = 2 / (2 + inner)
            q = q + alpha * (out - q)

    return q

def fugal(g1: np.ndarray, g2: np.ndarray, my: float = 1.5) -> np.ndarray:
    f1, f2 = extract_features(g1), extract_features(g2)
    distance = euclidian_distance(f1, f2)
    quasi_permutation = find_quasi_permutation(g1, g2, distance, my, 64)
    return [x for _, x in hungarian_algorithm(quasi_permutation)]

if __name__ == "__main__":
    generator = np.random.default_rng()
    vertex_count = 16

    g1 = create_random_adjacency_matrix(vertex_count, generator)
    g2, permutation = permute_adjacency_matrix(g1, generator)

    result = fugal(g1, g2)

    print("accuracy:", np.mean(result == permutation))