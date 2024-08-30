from dataclasses import dataclass
import numpy as np
import networkx as nx


def refill_edges(edges: np.array, dimension: int, edge_amount: int, generator: np.random.Generator) -> np.array:
    if edge_amount == 0:
        return edges

    # Set of all edges to check for duplicates.
    edge_set = {tuple(row) for row in np.sort(edges).tolist()}
    new_edges = []

    while len(new_edges) < edge_amount:
        new_edge = generator.integers(0, dimension, size=2)
        sorted_new_edge = tuple(np.sort(new_edge).tolist())

        if not (sorted_new_edge in edge_set) and new_edge[0] != new_edge[1]:
            edge_set.add(sorted_new_edge)
            new_edges.append(new_edge)

    return np.append(edges, new_edges, axis=0)


def remove_edges(edges: np.array, noise: float, generator: np.random.Generator):
    bin_count = np.bincount(edges.flatten())
    rows_to_delete = []
    for i, edge in enumerate(edges):
        if generator.random() < noise:
            e, f = edge
            if bin_count[e] > 1 and bin_count[f] > 1:
                bin_count[e] -= 1
                bin_count[f] -= 1
                rows_to_delete.append(i)
    return np.delete(edges, rows_to_delete, axis=0)


@dataclass
class GeneratedGraph:
    source_edges: np.array
    target_edges: np.array
    source_mapping: np.array
    target_mapping: np.array


def generate_graphs(
    graph: nx.Graph, generator: np.random.Generator, source_noise=0.0, target_noise=0.0, refill=False,
) -> GeneratedGraph:
    source_edges = np.array(graph.edges)

    if (np.amin(source_edges) != 0):
        source_edges = source_edges - np.amin(source_edges)

    dimension = np.amax(source_edges) + 1
    edge_count = source_edges.shape[0]

    source_permutation = np.arange(dimension)
    target_permutation = generator.permutation(dimension)

    source_mapping = source_permutation[target_permutation.argsort()]
    target_mapping = target_permutation[source_permutation.argsort()]

    target_edges = source_mapping[source_edges]

    source_edges = remove_edges(source_edges, source_noise, generator)
    target_edges = remove_edges(target_edges, target_noise, generator)

    if refill:
        source_edges = refill_edges(
            source_edges, dimension, edge_count - source_edges.shape[0], generator)
        target_edges = refill_edges(
            target_edges, dimension, edge_count - target_edges.shape[0], generator)

    return GeneratedGraph(
        source_edges=source_edges,
        target_edges=target_edges,
        source_mapping=source_mapping,
        target_mapping=target_mapping,
    )
