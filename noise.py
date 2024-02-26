import networkx as nx
import numpy as np

def remove_edges(
    G: nx.Graph,
    noise_level: float,
    generator: np.random.Generator,
) -> nx.Graph:
    edges = np.array(G.edges)
    retain_mask = generator.random(len(edges)) >= noise_level
    return nx.Graph(edges[retain_mask].tolist())

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