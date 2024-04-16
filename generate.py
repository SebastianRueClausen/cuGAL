import numpy as np
import networkx as nx


def refill_edges(edges, n, amount):
    if amount == 0:
        return edges

    ee = {tuple(row) for row in np.sort(edges).tolist()}

    new_e = []
    check = 0

    while len(new_e) < amount:
        _e = np.random.randint(n, size=2)
        _ee = tuple(np.sort(_e).tolist())

        check += 1

        if not (_ee in ee) and _e[0] != _e[1]:
            ee.add(_ee)
            new_e.append(_e)
            check = 0

    return np.append(edges, new_e, axis=0)


def remove_edges(edges, noise, no_disc=True, until_connected=False):
    ii = 0
    while True:
        ii += 1

        if no_disc:
            bin_count = np.bincount(edges.flatten())
            rows_to_delete = []
            for i, edge in enumerate(edges):
                if np.random.sample(1)[0] < noise:
                    e, f = edge
                    if bin_count[e] > 1 and bin_count[f] > 1:
                        bin_count[e] -= 1
                        bin_count[f] -= 1
                        rows_to_delete.append(i)
            new_edges = np.delete(edges, rows_to_delete, axis=0)
        else:
            new_edges = edges[np.random.sample(edges.shape[0]) >= noise]

        graph = nx.Graph(new_edges.tolist())
        graph_cc = len(max(nx.connected_components(graph), key=len))
        graph_connected = graph_cc == np.amax(edges) + 1

        if graph_connected or not until_connected:
            break

    return new_edges


def generate_graphs(graph: nx.Graph, source_noise=0.0, target_noise=0.0, refill=False):
    source_edges = np.array(graph.edges)

    if (np.amin(source_edges) != 0):
        source_edges = source_edges-np.amin(source_edges)

    n = np.amax(source_edges) + 1
    edge_count = source_edges.shape[0]

    permutations = np.array((
        np.arange(n),
        np.random.permutation(n)
    ))

    mappings = (
        permutations[:, permutations[1].argsort()][0],
        permutations[:, permutations[0].argsort()][1]
    )

    target_edges = mappings[0][source_edges]

    source_edges = remove_edges(source_edges, source_noise)
    target_edges = remove_edges(target_edges, target_noise)

    if refill:
        source_edges = refill_edges(
            source_edges, n, edge_count - source_edges.shape[0])
        target_edges = refill_edges(
            target_edges, n, edge_count - target_edges.shape[0])

    return source_edges, target_edges,  mappings
