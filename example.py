from dataclasses import dataclass, fields
import numpy as np
from cugal.pred import cugal
from cugal.config import Config, SinkhornMethod
from cugal.profile import Phase, Profile, write_phases_as_csv, plot_phases
import matplotlib.pyplot as plt
import networkx as nx
import metrics as metrics
import torch
import torch.cuda
import torch.backends.mps
import generate
from fugal.pred import fugal


def select_device() -> str:
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda:0'
    elif torch.backends.mps.is_available():
        device = 'mps'
    print("selected device:", device)
    return device


cpu_config = Config()

gpu_log_config = Config(
    device=select_device(),
    sinkhorn_method=SinkhornMethod.LOG,
    dtype=torch.float16,
)


@dataclass
class Experiment:
    config: Config
    source: nx.Graph
    target: nx.Graph | None = None
    source_noise: float = 0
    target_noise: float = 0
    refill_edges: bool = True
    generator: np.random.Generator = np.random.default_rng()


@dataclass
class Result:
    ics: float
    ec: float
    sss: float
    accuracy: float
    profile: Profile

    def __str__(self) -> str:
        metrics = [self.ics, self.ec, self.sss, self.accuracy]
        names = [
            "Induced Conserved Structure (ICS)",
            "Edge Correctness (EC)",
            "Symmetric Substructure Score (SSS)",
            "Accuracy",
        ]
        column_width = max(len(name) for name in names)
        return "\n".join(f"{n:<{column_width}} {m}" for n, m in zip(names, metrics))


def plot_results(experiments: list[Experiment], results: list[Result], x_axis_field: str):
    x = np.arange(len(experiments))
    for field in fields(Result):
        plt.plot(x, [getattr(result, field.name)
                 for result in results], label=field.name)
    x_labels = [getattr(experiment, x_axis_field)
                for experiment in experiments]
    if all(isinstance(label, float) for label in x_labels):
        x_labels = ["{:.2f}".format(label) for label in x_labels]
    plt.xticks(x, x_labels)
    plt.xlabel(x_axis_field)
    plt.legend()
    plt.show()


def add_missing_nodes(graph: nx.Graph, node_count: int):
    for i in set(range(node_count)) - set(graph.nodes()):
        graph.add_node(i)


def test(experiment: Experiment, use_fugal=False) -> Result:
    if experiment.target is not None:
        assert experiment.source.number_of_nodes() == experiment.target.number_of_nodes()

    if experiment.target is None:
        source, target, (source_mapping, _) = generate.generate_graphs(
            G=experiment.source,
            source_noise=experiment.target_noise,
            target_noise=experiment.target_noise,
            refill=experiment.refill_edges,
        )
        source, target = nx.from_edgelist(source), nx.from_edgelist(target)
    else:
        source, target = experiment.source, experiment.target
        source_mapping = np.arange(source.number_of_nodes())

    profile = Profile()

    if use_fugal:
        mapping = fugal(source, target, experiment.config.mu,
                        experiment.config.sinkhorn_iterations)
    else:
        _, mapping = cugal(source, target, experiment.config, profile)

    mapping = [x for _, x in mapping]

    A1 = nx.to_numpy_array(source)
    A2 = nx.to_numpy_array(target)

    assert A1.shape == A2.shape

    ics = metrics.induced_conserved_structure(A1, A2, source_mapping, mapping)
    ec = metrics.edge_correctness(A1, A2, source_mapping, mapping)
    sss = metrics.symmetric_substructure(A1, A2, source_mapping, mapping)
    accuracy = np.mean(mapping == source_mapping)

    return Result(ics, ec, sss, accuracy, profile)


def newmann_watts_graph(
    node_count: int,
    node_degree: int,
    rewriting_prob: float,
) -> nx.Graph:
    return nx.newman_watts_strogatz_graph(node_count, node_degree, rewriting_prob)


def multi_magna_graph() -> tuple[nx.Graph, nx.Graph]:
    f1 = open("data/yeast0_Y2H1.txt", "r")
    f2 = open("data/yeast10_Y2H1.txt", "r")

    nodes1 = [tuple(map(int, n.split())) for n in f1.read().split("\n") if n]
    nodes2 = [tuple(map(int, n.split())) for n in f2.read().split("\n") if n]

    n1 = max([max(n) for n in nodes1])
    n2 = max([max(n) for n in nodes2])

    assert n1 == n2

    f1.close()
    f2.close()

    A1 = np.zeros((n1, n1))
    for i, j in nodes1:
        A1[i - 1, j - 1] = 1
        A1[j - 1, i - 1] = 1

    A2 = np.zeros((n1, n1))
    for i, j in nodes2:
        A2[i - 1, j - 1] = 1
        A2[j - 1, i - 1] = 1

    return nx.from_numpy_array(A1), nx.from_numpy_array(A2)


def multi_magna_experiment(device: str) -> Experiment:
    config = Config(
        device=device,
        sinkhorn_method=SinkhornMethod.MIX,
        use_sparse_adjacency=True,
        use_sinkhorn_cache=True,
        dtype=torch.float32,
        mu=2.0,
    )
    return Experiment(config, *multi_magna_graph(), target_noise=0.05)


def wiki_graph() -> nx.Graph:
    import json

    with open("data/graph.json", "r") as file:
        articles = json.loads(file.read())

    graph = nx.Graph()

    for _, article in articles.items():
        index = article['index']
        for neighbor in article['neighbors']:
            graph.add_edge(index, articles[neighbor]['index'])

    return graph


def wiki_experiment(device: str) -> Experiment:
    config = Config(
        device=device,
        sinkhorn_method=SinkhornMethod.MIX,
        use_sparse_adjacency=True,
        use_sinkhorn_cache=True,
        dtype=torch.float32,
        sinkhorn_threshold=1e-3,
        iter_count=15,
        mu=2.0,
    )
    return Experiment(config, wiki_graph(), target_noise=0.0)


def newmann_watts_experiment(config: Config, source_noise: float, size: int) -> Experiment:
    graph = newmann_watts_graph(
        node_count=size, node_degree=7, rewriting_prob=0.1)
    return Experiment(config, graph, source_noise=source_noise)


def replicate_figure_4(config: Config):
    config = Config(mu=2, sinkhorn_method=SinkhornMethod.LOG,
                    device=select_device(), dtype=torch.float16)
    noises = np.linspace(0, 0.25, num=6)
    experiments = [newmann_watts_experiment(
        config, source_noise=noise) for noise in noises]
    results = list(map(test, experiments))
    plot_results(experiments, results, "source_noise")


def compare_against_official():
    node_count = 1024
    graph = newmann_watts_graph(
        node_count=node_count, node_degree=7, rewriting_prob=0.01)
    config = Config(sinkhorn_method=SinkhornMethod.STANDARD)

    cugal_mapping = cugal(graph, graph, config)
    cugal_mapping = [x for _, x in cugal_mapping]

    fugal_mapping = fugal(graph, graph, config.mu, config.iter_count)
    fugal_mapping = [x for _, x in fugal_mapping]

    cugal_accuracy = np.mean(np.arange(node_count) == cugal_mapping)
    fugal_accuracy = np.mean(np.arange(node_count) == fugal_mapping)

    print("accuracy:", cugal_accuracy, "official accuracy:", fugal_accuracy)


def newmann_watts_benchmark():
    sizes = [128, 256, 512, 1024, 2048, 4096]
    profiles = []
    for _, size in enumerate(sizes):
        result = test(newmann_watts_experiment(Config(use_sparse_adjacency=True, sinkhorn_method=SinkhornMethod.MIX,
                                                      dtype=torch.float32, device="cuda"), 0.0, size))
        profiles.append(result.profile)
    print(profiles[-1].phase_times)
    plot_phases(profiles, sizes)


if __name__ == "__main__":
    # replicate_figure_4(gpu_log_config)
    # compare_against_official()
    # newmann_watts_benchmark()
    # print(test(multi_magna_experiment(select_device())))
    print(test(wiki_experiment(select_device())))
