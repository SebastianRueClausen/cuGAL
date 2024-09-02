from dataclasses import dataclass
import dataclasses
from enum import Enum
import io
import subprocess
import numpy as np
import cugal
from cugal.pred import cugal
from cugal.config import Config
from cugal.profile import Profile, TimeStamp
import networkx as nx
import requests
import gzip
import datetime
from fugal.pred import fugal
import json

def refill_edges(edges: np.array, dimension: int, edge_amount: int, generator: np.random.Generator) -> np.array:
    if edge_amount == 0: return edges
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


@dataclass
class Algorithm:
    config: Config
    use_fugal: bool

    def to_dict(self) -> dict:
        return {'use_fugal': self.use_fugal, 'config': self.config.to_dict()}

    @classmethod
    def from_dict(cls, dict: dict):
        return cls(config=Config.from_dict(dict['config']), use_fugal=dict['use_fugal'])


def create_graph_from_str(file: str) -> nx.Graph:
    edges = [tuple(map(int, line.split()))
             for line in file.split("\n") if not line.startswith('#')]
    return nx.from_edgelist([edge for edge in edges if len(edge) == 2])


class GraphKind(Enum):
    CA_HEP = "CA_HEP"
    NEWMAN_WATTS = "NEWMAN_WATTS"
    LOBSTER = "LOBSTER"


@dataclass
class Graph:
    kind: GraphKind
    parameters: dict

    def get(self, generator: np.random.Generator) -> tuple[nx.Graph, nx.Graph | None]:
        match self.kind:
            case GraphKind.CA_HEP:
                response = requests.get(
                    'https://snap.stanford.edu/data/ca-HepTh.txt.gz')
                with gzip.GzipFile(fileobj=io.BytesIO(response.content)) as gz_file:
                    file_content = gz_file.read().decode('utf-8')
                return create_graph_from_str(file_content), None
            case GraphKind.NEWMAN_WATTS:
                return nx.newman_watts_strogatz_graph(**self.parameters, seed=generator), None
            case GraphKind.LOBSTER:
                return nx.random_lobster(**self.parameters, seed=generator), None
            
    def to_dict(self) -> dict:
        return { 'kind': self.kind.value, 'parameters': self.parameters }
    
    @classmethod
    def from_dict(cls, dict: dict):
        dict['kind'] = GraphKind[dict['kind']]
        return cls(**dict)

@dataclass
class NoiseLevel:
    source_noise: float
    target_noise: float
    refill_edges: bool


def generate_graph(
    source: nx.Graph,
    target: nx.Graph | None,
    generator: np.random.Generator,
    noise_level: NoiseLevel,
) -> tuple[nx.Graph, nx.Graph | None, np.array]:
    if target is None:
        generated_graph = generate_graphs(
            source,
            generator,
            source_noise=noise_level.target_noise,
            target_noise=noise_level.target_noise,
            refill=noise_level.refill_edges,
        )
        source = nx.from_edgelist(generated_graph.source_edges)
        target = nx.from_edgelist(generated_graph.target_edges)
        mapping = generated_graph.source_mapping
    else:
        # FIXME
        mapping = np.arange(source.number_of_nodes())
    return source, target, mapping


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
            induced_conserved_structure(source_adjacency, target_adjacency, answer, mapping),
            edge_correctness(source_adjacency, target_adjacency, answer, mapping),
            symmetric_substructure(source_adjacency, target_adjacency, answer, mapping),
            alignment_accuracy(answer, mapping),
            profile,
        )

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
    seed: int = np.random.randint(0xffffffff)

    def to_dict(self) -> dict:
        dict = dataclasses.asdict(self)
        dict['algorithms'] = [algorithm.to_dict()
                              for algorithm in self.algorithms]
        dict['graphs'] = [graph.to_dict() for graph in self.graphs]
        dict['noise_levels'] = [dataclasses.asdict(
            level) for level in self.noise_levels]
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
        generator = np.random.default_rng(seed=self.seed)
        results = []
        for graph in self.graphs:
            source_graph, target_graph = graph.get(generator)
            graph_results = []
            for noise_level in self.noise_levels:
                noise_results = []
                source, target, mapping = generate_graph(source_graph, target_graph, generator, noise_level)
                for algorithm in self.algorithms:
                    profile = Profile()
                    if algorithm.use_fugal:
                        start_time = TimeStamp('cpu')
                        _, answer = fugal(source, target, algorithm.config.mu, algorithm.config.iter_count, algorithm.config, profile)
                        profile.time = TimeStamp('cpu').elapsed_seconds(start_time)
                    else:
                        _, answer = cugal(source, target, algorithm.config, profile)
                    noise_results.append(Result.calculate(
                        profile,
                        nx.to_numpy_array(source),
                        nx.to_numpy_array(target),
                        np.array([x for _, x in answer]),
                        mapping,
                    ))
                graph_results.append(noise_results)
            results.append(graph_results)
        return ExperimentResults.from_results(self, results)


@dataclass
class ExperimentResults:
    experiment: Experiment
    commit: str
    time: str
    # For each graph, for each noise level, for each algorithm.
    results: list[list[list[Result]]]

    def dump(self, folder: str):
        with open(folder + "/" + self.name() + ".json", "w") as f: json.dump(self.to_dict(), f, indent=4)

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
        dict['results'] = [[[Result.from_dict(result) for result in b] for b in a] for a in dict['results']]
        return cls(**dict)
    
    def name(self) -> str:
        return '-'.join(map(lambda graph: graph.kind.value, self.experiment.graphs)) + '-' + self.time.replace(' ', '-')

    def to_dict(self) -> dict:
        dict = dataclasses.asdict(self)
        dict['experiment'] = self.experiment.to_dict()
        dict['results'] = [[[result.to_dict() for result in b] for b in a]
                           for a in self.results]
        return dict