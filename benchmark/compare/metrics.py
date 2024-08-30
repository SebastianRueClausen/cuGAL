import numpy as np


def edge_correctness(
    source_adjacency: np.array,
    target_adjacency: np.array,
    source_mapping: np.array,
    target_mapping: np.array,
) -> float:
    adj1 = source_adjacency[source_mapping][:, source_mapping]
    adj2 = target_adjacency[target_mapping][:, target_mapping]
    return np.sum(adj1 + adj2 == 2) / np.sum(source_adjacency == 1)


def induced_conserved_structure(
    source_adjacency: np.array,
    target_adjacency: np.array,
    source_mapping: np.array,
    target_mapping: np.array,
) -> float:
    adj1 = source_adjacency[source_mapping][:, source_mapping]
    adj2 = target_adjacency[target_mapping][:, target_mapping]
    return np.sum(adj1 + adj2 == 2) / np.sum(adj2 == 1)


def symmetric_substructure(
    source_adjacency: np.array,
    target_adjacency: np.array,
    source_mapping: np.array,
    target_mapping: np.array,
) -> float:
    adj1 = source_adjacency[source_mapping][:, source_mapping]
    adj2 = target_adjacency[target_mapping][:, target_mapping]
    intersection = np.sum(adj1 + adj2 == 2)
    return intersection / (np.sum(source_adjacency == 1) + np.sum(adj2 == 1) - intersection)


def accuracy(ground_truth: np.array, mapping: np.array) -> float:
    return np.mean(mapping == ground_truth)
