import numpy as np
from hungarian import hungarian_algorithm
from sinkhorn_knopp import sinkhorn_knopp as skp
from sinkhorn import sinkhorn

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

    div_mask = vertex_1st_degrees!=0

    mean_neighbour_degress = np.divide(
        vertex_1st_degrees @ adjacency,
        vertex_1st_degrees,
        where=div_mask,
    )

    mean_neighbour_clustering = np.divide(
        vertex_clustering @ adjacency,
        vertex_1st_degrees,
        where=div_mask,
    )

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
    # The maximum is only due to precision issues, where some entries might be -0.0000007.
    return np.sqrt(np.maximum(p1 + p2 + p3, 0.0))

def find_quasi_permutation(
    A: np.ndarray,
    B: np.ndarray,
    D: np.ndarray,
    mu: float,
    iteration_count: int,
) -> np.ndarray:
    vertex_count, _ = A.shape
    ones = np.ones(vertex_count)
    Q = np.outer(ones, ones) / vertex_count
    J = np.ones(shape=A.shape)

    f_deriv = lambda P: -A @ P @ B.T - A.T @ P @ B + mu * D
    g_deriv = lambda P: J - 2*P

    for lam in range(iteration_count):
        for it in range(1, 11):
            grad = f_deriv(Q) + lam * g_deriv(Q)
            q_it = sinkhorn(ones, ones, grad, reg=1.0, maxIter=500, stopThr=1e-3)
            alpha = 2 / (2.0 + it)
            Q += alpha * (q_it - Q)

    return Q

def fugal(g1: np.ndarray, g2: np.ndarray, mu: float = 2.0) -> np.ndarray:
    f1, f2 = extract_features(g1), extract_features(g2)
    distance = euclidian_distance(f1, f2)
    quasi_permutation = find_quasi_permutation(g1, g2, distance, mu, 15)
    return [x for _, x in hungarian_algorithm(quasi_permutation)]

if __name__ == "__main__":
    generator = np.random.default_rng()
    vertex_count = 128

    means = []

    for _ in range(1):
        g1 = create_random_adjacency_matrix(vertex_count, generator, connectivity=0.1)
        g2, permutation = permute_adjacency_matrix(g1, generator)

        result = fugal(g1, g2)
        means.append(np.mean(result == permutation))

    print("accuracy:", np.mean(means))