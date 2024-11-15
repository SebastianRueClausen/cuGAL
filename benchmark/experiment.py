from dataclasses import dataclass
import dataclasses
from enum import Enum
import io
import subprocess
from typing import Iterable
import numpy as np
import cugal
from cugal.pred import cugal
from cugal.config import Config
from cugal.profile import Profile, TimeStamp
import networkx as nx
import requests
import gzip
import datetime
import json
import scipy.sparse as sps
import FUGAL.Fugal as Fugal
from typing import Self


@dataclass(frozen=True)
class NoiseLevel:
    source_noise: float
    target_noise: float
    refill_edges: bool

    def __str__(self):
        return str(dataclasses.asdict(self))

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def refill_edges(edges: np.array, dimension: int, edge_amount: int, generator: np.random.Generator) -> np.array:
    """Randomly insert `edge_amount` of edges."""
    if edge_amount == 0:
        return edges
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
    """Randomly remove edges with `noise` chance."""
    bin_count = np.bincount(edges.flatten())
    rows_to_delete = []
    for i, (e, f) in enumerate(edges):
        if generator.random() < noise:
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


def permute_edges(edges: np.array, dimension: int) -> tuple[np.array, np.array, np.array]:
    source_permutation = np.arange(dimension)
    target_permutation = np.random.RandomState(seed=0).permutation(dimension)
    source_mapping = source_permutation[target_permutation.argsort()]
    target_mapping = target_permutation[source_permutation.argsort()]
    assert np.all(target_permutation[source_mapping] == source_permutation)
    assert np.all(source_permutation[target_mapping] == target_permutation)
    return source_mapping[edges], source_mapping, target_mapping


def add_synthetic_noise(
    graph: nx.Graph, generator: np.random.Generator, noise_level: NoiseLevel,
) -> GeneratedGraph:
    source_edges = np.array(graph.edges)
    if (np.amin(source_edges) != 0):
        source_edges = source_edges - np.amin(source_edges)
    dimension = np.amax(source_edges) + 1

    # Permute the edges of the source graph to create a target graph
    target_edges, source_mapping, target_mapping = permute_edges(
        source_edges, dimension)

    # Remove edges according to the noise level.
    edge_count = source_edges.shape[0]
    source_edges = remove_edges(
        source_edges, noise_level.source_noise, generator)
    target_edges = remove_edges(
        target_edges, noise_level.target_noise, generator)

    # Refill edges according to the noise level.
    if noise_level.refill_edges:
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


def edge_correctness(source_adjacency: np.array, target_adjacency: np.array, source_mapping: np.array, target_mapping: np.array) -> float:
    adj1 = source_adjacency[source_mapping][:, source_mapping]
    adj2 = target_adjacency[target_mapping][:, target_mapping]
    return np.sum(adj1 + adj2 == 2) / np.sum(source_adjacency == 1)


def induced_conserved_structure(source_adjacency: np.array, target_adjacency: np.array, source_mapping: np.array, target_mapping: np.array) -> float:
    adj1 = source_adjacency[source_mapping][:, source_mapping]
    adj2 = target_adjacency[target_mapping][:, target_mapping]
    return np.sum(adj1 + adj2 == 2) / np.sum(adj2 == 1)


def symmetric_substructure(source_adjacency: np.array, target_adjacency: np.array, source_mapping: np.array, target_mapping: np.array) -> float:
    adj1 = source_adjacency[source_mapping][:, source_mapping]
    adj2 = target_adjacency[target_mapping][:, target_mapping]
    intersection = np.sum(adj1 + adj2 == 2)
    return intersection / (np.sum(source_adjacency == 1) + np.sum(adj2 == 1) - intersection)


def alignment_accuracy(ground_truth: np.array, mapping: np.array) -> float:
    return np.mean(mapping == ground_truth)


@dataclass(frozen=True)
class Algorithm:
    config: Config
    use_fugal: bool

    def to_dict(self) -> dict:
        return {'use_fugal': self.use_fugal, 'config': self.config.to_dict()}

    def __str__(self):
        name = 'Fugal' if self.use_fugal else 'Cugal'
        return name + ' ' + str(self.config.to_dict())

    @classmethod
    def from_dict(cls, dict: dict):
        return cls(config=Config.from_dict(dict['config']), use_fugal=dict['use_fugal'])


def create_graph_from_str(file: str) -> nx.Graph:
    """
    Creates a graph from a string.
    The string should be in the format of the SNAP dataset.
    I.e. each line should contain two integers separated by a space.
    """
    edges = [tuple(map(int, line.split()))
             for line in file.split("\n") if not line.startswith('#')]
    return nx.from_edgelist([edge for edge in edges if len(edge) == 2])


class GraphKind(Enum):
    # Real-world graphs
    CA_HEP = "CA_HEP"
    CA_ERDOS = "CA_ERDOS"
    CA_GRQC = "CA_GRQC"
    CA_NETSCIENCE = "CA_NETSCIENCE"
    BIO_DMELA = "BIO_DMELA"
    BIO_CELEGANS = "BIO_CELEGANS"
    IN_ARENAS = "IN_ARENAS"
    INF_POWER = "INF_POWER"
    INF_EUROROAD = "INF_EUROROAD"
    SOC_FACEBOOK = "SOC_FACEBOOK"
    SOC_HAMSTERSTER = "SOC_HAMSTERSTER"
    SOCFB_BOWDOIN47 = "SOCFB_BOWDOIN47"
    SOCFB_HAMILTON46 = "SOCFB_HAMILTON46"
    SOCFB_HAVERFORD76 = "SOCFB_HAVERFORD76"
    SOCFB_SWARTHMORE42 = "SOCFB_SWARTHMORE42"

    # Synthetic graphs
    NEWMAN_WATTS = "NEWMAN_WATTS"
    LOBSTER = "LOBSTER"

    # Predefined graphs
    PREDEFINED_GRAPHS = "PREDEFINED_GRAPHS"

    # Averaged results (only used for analysis)
    AVERAGED = "AVERAGED"


@dataclass(frozen=True)
class Graph:
    kind: GraphKind
    parameters: dict

    def get(self, generator: np.random.Generator) -> tuple[nx.Graph, nx.Graph | None]:
        match self.kind:
            case GraphKind.NEWMAN_WATTS:
                return nx.newman_watts_strogatz_graph(**self.parameters, seed=generator), None
            case GraphKind.LOBSTER:
                return nx.random_lobster(**self.parameters, seed=generator), None
            case GraphKind.PREDEFINED_GRAPHS:
                # Load graphs from provided files
                graph_file_1 = open(self.parameters['source_file'], 'r')
                graph_file_2 = open(self.parameters['target_file'], 'r')
                file_content_1 = graph_file_1.read()
                file_content_2 = graph_file_2.read()

                return create_graph_from_str(file_content_1), \
                    create_graph_from_str(file_content_2)

            case GraphKind.CA_HEP:
                response = requests.get(
                    'https://snap.stanford.edu/data/ca-HepTh.txt.gz')
                with gzip.GzipFile(fileobj=io.BytesIO(response.content)) as gz_file:
                    file_content = gz_file.read().decode('utf-8')
                return create_graph_from_str(file_content), None
            case GraphKind.IN_ARENAS:
                graph_file = open("data/in-arenas.txt", 'r')
                file_content = graph_file.read()
                return create_graph_from_str(file_content), None
            case GraphKind.INF_POWER:
                graph_file = open("data/inf-power.txt", 'r')
                file_content = graph_file.read()
                return create_graph_from_str(file_content), None
            case GraphKind.BIO_DMELA:
                graph_file = open("data/bio-dmela.txt", 'r')
                file_content = graph_file.read()
                return create_graph_from_str(file_content), None
            case GraphKind.BIO_CELEGANS:
                graph_file = open("data/bio-celegans.txt", 'r')
                file_content = graph_file.read()
                return create_graph_from_str(file_content), None
            case GraphKind.CA_NETSCIENCE:
                graph_file = open("data/ca-netscience.txt", 'r')
                file_content = graph_file.read()
                return create_graph_from_str(file_content), None
            case GraphKind.CA_ERDOS:
                graph_file = open("data/ca-Erdos992.txt", 'r')
                file_content = graph_file.read()
                return create_graph_from_str(file_content), None
            case GraphKind.CA_GRQC:
                graph_file = open("data/ca-GrQc.txt", 'r')
                file_content = graph_file.read()
                return create_graph_from_str(file_content), None
            case GraphKind.INF_EUROROAD:
                graph_file = open("data/inf-euroroad.txt", 'r')
                file_content = graph_file.read()
                return create_graph_from_str(file_content), None
            case GraphKind.SOC_FACEBOOK:
                graph_file = open("data/socfb-Bowdoin47.txt", 'r')
                file_content = graph_file.read()
                return create_graph_from_str(file_content), None
            case GraphKind.SOC_HAMSTERSTER:
                graph_file = open("data/socfb-Hamilton46.txt", 'r')
                file_content = graph_file.read()
                return create_graph_from_str(file_content), None
            case GraphKind.SOCFB_BOWDOIN47:
                graph_file = open("data/socfb-Bowdoin47.txt", 'r')
                file_content = graph_file.read()
                return create_graph_from_str(file_content), None
            case GraphKind.SOCFB_HAMILTON46:
                graph_file = open("data/socfb-Hamilton46.txt", 'r')
                file_content = graph_file.read()
                return create_graph_from_str(file_content), None
            case GraphKind.SOCFB_HAVERFORD76:
                graph_file = open("data/socfb-Haverford76.txt", 'r')
                file_content = graph_file.read()
                return create_graph_from_str(file_content), None
            case GraphKind.SOCFB_SWARTHMORE42:
                graph_file = open("data/socfb-Swarthmore42.txt", 'r')
                file_content = graph_file.read()
                return create_graph_from_str(file_content), None

    def to_dict(self) -> dict:
        return {'kind': self.kind.value, 'parameters': self.parameters}

    def __str__(self):
        return self.kind.value + ' ' + str(self.parameters)

    @classmethod
    def from_dict(cls, dict: dict):
        dict['kind'] = GraphKind[dict['kind']]
        return cls(**dict)


def generate_graph(
    source: nx.Graph,
    target: nx.Graph | None,
    generator: np.random.Generator,
    noise_level: NoiseLevel,
) -> tuple[nx.Graph, nx.Graph | None, np.ndarray, np.ndarray]:
    if target is None:
        generated_graph = add_synthetic_noise(source, generator, noise_level)
        source = nx.from_edgelist(generated_graph.source_edges)
        target = nx.from_edgelist(generated_graph.target_edges)
        return source, target, generated_graph.source_mapping,  generated_graph.target_mapping
    else:
        source_edges = np.array(source.edges)
        if (np.amin(source_edges) != 0):
            source_edges = source_edges - np.amin(source_edges)
        target_edges = np.array(target.edges)
        if (np.amin(target_edges) != 0):
            target_edges = target_edges - np.amin(target_edges)
        target_edges, source_mapping, target_mapping = permute_edges(
            target_edges, np.amax(target_edges) + 1)
        source = nx.from_edgelist(source_edges)
        target = nx.from_edgelist(target_edges)
        return source, target, source_mapping, target_mapping


@dataclass
class Result:
    ics: float
    ec: float
    sss: float
    accuracy: float
    profile: Profile

    @classmethod
    def calculate(cls, profile: Profile, source_adjacency: np.array, target_adjacency: np.array, answer: np.array, mapping: np.array):
        return cls(
            induced_conserved_structure(
                source_adjacency, target_adjacency, answer, mapping),
            edge_correctness(source_adjacency,
                             target_adjacency, answer, mapping),
            symmetric_substructure(
                source_adjacency, target_adjacency, answer, mapping),
            alignment_accuracy(answer, mapping),
            profile,
        )

    @staticmethod
    def average(results: list[Self]) -> Self:
        ics = sum(result.ics for result in results) / len(results)
        ec = sum(result.ec for result in results) / len(results)
        sss = sum(result.sss for result in results) / len(results)
        accuracy = sum(result.accuracy for result in results) / len(results)
        profile = Profile.average([result.profile for result in results])
        return Result(ics, ec, sss, accuracy, profile)

    def __str__(self) -> str:
        metrics = [self.ics, self.ec, self.sss,
                   self.accuracy, self.profile.time]
        names = [
            "Induced Conserved Structure (ICS)",
            "Edge Correctness (EC)",
            "Symmetric Substructure Score (SSS)",
            "Accuracy",
            "Time (seconds)",
        ]
        column_width = max(len(name) for name in names)
        return "\n".join(f"{n:<{column_width}} {m}" for n, m in zip(names, metrics))

    def to_dict(self) -> dict:
        dict = dataclasses.asdict(self)
        dict['profile'] = self.profile.to_dict()
        return dict

    @classmethod
    def from_dict(cls, dict: dict):
        dict['profile'] = Profile.from_dict(dict['profile'])
        return cls(**dict)


def get_last_commit() -> str:
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return ""


@dataclass
class Experiment:
    algorithms: list[Algorithm]
    graphs: list[Graph]
    noise_levels: list[NoiseLevel]
    debug: bool = False
    seed: int | None = None
    save_alignment: bool = False
    num_runs: int = 1

    def to_dict(self) -> dict:
        dict = dataclasses.asdict(self)
        dict['algorithms'] = [algorithm.to_dict()
                              for algorithm in self.algorithms]
        dict['graphs'] = [graph.to_dict()
                          for graph in self.graphs]
        dict['noise_levels'] = [dataclasses.asdict(level)
                                for level in self.noise_levels]
        return dict

    @classmethod
    def from_dict(cls, dict: dict):
        dict['algorithms'] = [Algorithm.from_dict(
            algorithm) for algorithm in dict["algorithms"]]
        dict['graphs'] = [Graph.from_dict(graph) for graph in dict['graphs']]
        dict['noise_levels'] = [NoiseLevel(**level)
                                for level in dict['noise_levels']]
        return cls(**dict)

    def run(self):
        if self.seed is None:
            self.seed = np.random.randint(0xffffffff)
        assert len(self.algorithms) > 0, "no algorithms provided"
        assert len(self.noise_levels) > 0, "no noise levels provided"
        assert len(self.graphs) > 0, "no graphs provided"

        # Make numpy random generator with seed
        generator = np.random.default_rng(seed=self.seed)

        results = []

        # Run on each graph (Type Graph) provided for the experiment
        for graph in self.graphs:
            print(f"Running on {str(graph)}")

            source_graph, target_graph = graph.get(generator)
            graph_results = []

            for noise_level in self.noise_levels:
                print(f"Running with noise level {str(noise_level)}")

                noise_results = []
                sources, targets, source_mappings = [], [], []
                for i in range(self.num_runs):
                    source, target, source_mapping, _ = generate_graph(
                        source_graph, target_graph, generator, noise_level)
                    sources.append(source)
                    targets.append(target)
                    source_mappings.append(source_mapping)

                for algorithm in self.algorithms:
                    print(f"Running with algorithm {str(algorithm)}")
                    # Run multiple times to get average results
                    run_results = []
                    for i in range(self.num_runs):
                        profile = Profile()
                        if algorithm.use_fugal:
                            start_time = TimeStamp('cpu')
                            _, answer = Fugal.main(
                                {"Src": edges_to_adjacency_matrix(np.array(source.edges),
                                                                  source_mappings[i].shape[0]),
                                 "Tar": edges_to_adjacency_matrix(np.array(target.edges),
                                                                  source_mappings[i].shape[0])},
                                algorithm.config.iter_count,
                                True, algorithm.config.mu
                            )
                            profile.time = TimeStamp(
                                'cpu').elapsed_seconds(start_time)
                        else:
                            _, answer = cugal(
                                sources[i], targets[i], algorithm.config, profile)

                        if self.save_alignment:
                            with open(str(datetime.datetime.now()) + ".txt", "w") as f:
                                f.write("\n".join(
                                    f"{x} {y}" for x, y in answer))

                        run_results.append(Result.calculate(
                            profile,
                            nx.to_numpy_array(sources[i]),
                            nx.to_numpy_array(targets[i]),
                            np.array([x for _, x in answer]),
                            source_mappings[i],
                        ))

                    noise_results.append(Result.average(run_results))
                graph_results.append(noise_results)
            results.append(graph_results)
        return ExperimentResults.from_results(self, results)


def edges_to_adjacency_matrix(edges: np.array, dimension: int):
    edge_count = edges.shape[0]
    G = sps.csr_matrix((np.ones(edge_count), edges.T),
                       shape=(dimension, dimension), dtype=np.int8)
    G += G.T
    G.data = G.data.clip(0, 1)
    return G.A


@dataclass
class ExperimentResults:
    experiment: Experiment
    commit: str
    time: str
    # For each graph, for each noise level, for each algorithm.
    results: list[list[list[Result]]]

    def dump(self, folder: str):
        with open(folder + "/" + self.name() + ".json", "w") as f:
            json.dump(self.to_dict(), f, indent=4)

    def all_results(self) -> Iterable[tuple[Graph, NoiseLevel, Algorithm, Result]]:
        for graph, graph_results in zip(self.experiment.graphs, self.results):
            for noise_level, noise_level_results in zip(self.experiment.noise_levels, graph_results):
                for algorithm, result in zip(self.experiment.algorithms, noise_level_results):
                    yield (graph, noise_level, algorithm, result)

    @classmethod
    def from_results(cls, experiment: Experiment, results: list[list[list[Result]]]):
        return ExperimentResults(
            experiment=experiment,
            results=results,
            time=str(datetime.datetime.now()),
            commit=get_last_commit(),
        )

    @classmethod
    def from_dict(cls, dict: dict):
        dict['experiment'] = Experiment.from_dict(dict['experiment'])
        dict['results'] = [
            [[Result.from_dict(result) for result in b] for b in a] for a in dict['results']]
        return cls(**dict)

    def name(self) -> str:
        return '-'.join(map(lambda graph: graph.kind.value, self.experiment.graphs)) + '-' + self.time.replace(' ', '-')

    def to_dict(self) -> dict:
        dict = dataclasses.asdict(self)
        dict['experiment'] = self.experiment.to_dict()
        dict['results'] = [[[result.to_dict() for result in b] for b in a]
                           for a in self.results]
        return dict

    def algorithms_descriptions(algorithms: list[Algorithm]) -> dict[Algorithm, str]:
        uses_both = False
        different_fields = set()
        for a in algorithms:
            for b in algorithms:
                if a.use_fugal != b.use_fugal:
                    uses_both = True
                a_dict, b_dict = a.config.to_dict(), b.config.to_dict()
                for field in a_dict.keys():
                    if str(a_dict[field]) != str(b_dict[field]):
                        different_fields.add(field)
        descriptions = {}
        for algorithm in algorithms:
            description = ""
            if uses_both:
                description += "FUGAL" if algorithm.use_fugal else "cuGAL"
            as_dict = algorithm.config.to_dict()
            for field in different_fields:
                if algorithm.use_fugal:
                    continue
                description += f"-{field}: {str(as_dict[field])}"
            descriptions[algorithm] = description
        return descriptions

    def graph_descriptions(graphs: list[Graph]) -> dict[Graph, str]:
        descriptions = {}
        different_fields = set()
        for graph in graphs:
            for other_graph in graphs:
                for field in graph.parameters.keys():
                    if graph.parameters[field] != other_graph.parameters[field]:
                        different_fields.add(field)
        for graph in graphs:
            description = graph.kind.value
            for field in different_fields:
                description += f"-{field}: {str(graph.parameters[field])}"
            descriptions[str(graph)] = description

        return descriptions
