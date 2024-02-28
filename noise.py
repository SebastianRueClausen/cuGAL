import networkx as nx
import numpy as np

def remove_edges(
    G: nx.Graph,
    noise_level: float,
    generator: np.random.Generator,
) -> nx.Graph:
    edges = np.array(G.edges)
    bin_count = np.bincount(edges.flatten())
    rows_to_delete = []
    for i, edge in enumerate(edges):
        if generator.random(1) < noise_level:
            e, f = edge
            if bin_count[e] > 1 and bin_count[f] > 1:
                bin_count[e] -= 1
                bin_count[f] -= 1
                rows_to_delete.append(i)
    new_edges = np.delete(edges, rows_to_delete, axis=0)
    return nx.Graph(new_edges.tolist())

def add_edges(
    G: nx.Graph,
    amount: int,
    generator: np.random.Generator,
) -> nx.Graph:
    n = G.number_of_nodes()
    target_edge_count = G.number_of_edges() + amount
    while G.number_of_edges() < target_edge_count:
        edge = generator.integers(low=0, high=n, size=2)
        if edge[0] == edge[1] or G.has_edge(edge[0], edge[1]): continue
        G.add_edge(edge[0], edge[1])
    return G