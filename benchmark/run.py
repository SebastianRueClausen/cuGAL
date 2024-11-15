from experiment import *
from cugal.config import SinkhornMethod, HungarianMethod, Config
import os
import torch
from dataclasses import replace


config = Config(
    device='cuda:1', 
    sinkhorn_method=SinkhornMethod.LOG,
    dtype=torch.float32,
    sinkhorn_threshold=0.2,
    sinkhorn_iterations=500,
    mu=2,
    iter_count=15,
    use_sparse_adjacency=True,
    sinkhorn_cache_size=1,
    frank_wolfe_threshold=0.2,
    frank_wolfe_iter_count=10,
    recompute_distance=True,
    hungarian_method=HungarianMethod.SCIPY,
    sinkhorn_regularization=1,
    dynamic_sinkhorn_regularization=True
    sinkhorn_scaling=1,
    )
mus = [0.1, 0.5, 1, 2]
experiment = Experiment(
    graphs=[
        #Graph(GraphKind.PREDEFINED_GRAPHS, {
        #    'source_file': 'data/MultiMagna/yeast0_Y2H1.txt',
        #    'target_file': f'data/MultiMagna/yeast{i}_Y2H1.txt',
        #}) for i in range(5, 26, 5)
        #Graph(GraphKind.NEWMAN_WATTS, {'n': 1000, 'k': i, 'p': 0.2}) for i in range(5, 26, 5)
        #Graph(GraphKind.NEWMAN_WATTS, {'n': 1000, 'k': 10, 'p': i}) for i in [0.1, 0.2, 0.3, 0.4, 0.5]
        #Graph(GraphKind.NEWMAN_WATTS, {'n': 1000, 'k': 10, 'p': 0.2}),
        Graph(GraphKind.BIO_DMELA, {}),
        Graph(GraphKind.CA_ERDOS, {}),
        Graph(GraphKind.CA_GRQC, {}),
        Graph(GraphKind.CA_NETSCIENCE, {}),
        Graph(GraphKind.IN_ARENAS, {}),
        Graph(GraphKind.INF_POWER, {}),
        Graph(GraphKind.INF_EUROROAD, {}),
        #Graph(GraphKind.SOC_FACEBOOK, {}),
        #Graph(GraphKind.SOC_HAMSTERSTER, {}),
        #Graph(GraphKind.SOCFB_BOWDOIN47, {}),
        #Graph(GraphKind.SOCFB_HAMILTON46, {}),
        #Graph(GraphKind.SOCFB_HAVERFORD76, {}),
        #Graph(GraphKind.SOCFB_SWARTHMORE42, {}),
    ],
    algorithms=np.array([
        #Algorithm(config, use_fugal=True),
        #[Algorithm(replace(config, mu=mu, use_fugal=False))                              for mu in mus],
        #[Algorithm(replace(config, sinkhorn_regularization=0.5, mu=mu), use_fugal=False) for mu in mus],
        #[Algorithm(replace(config, sinkhorn_regularization=0.1, mu=mu), use_fugal=False) for mu in mus],
        #[Algorithm(replace(config, sinkhorn_regularization=0.05, mu=mu), use_fugal=False)for mu in mus],
        Algorithm(replace(config, sinkhorn_scaling=32), use_fugal=False),
        Algorithm(replace(config, sinkhorn_scaling=48), use_fugal=False),
        Algorithm(replace(config, sinkhorn_scaling=64), use_fugal=False),
        Algorithm(replace(config, sinkhorn_scaling=96), use_fugal=False),
        Algorithm(replace(config, sinkhorn_scaling=128), use_fugal=False),
        Algorithm(replace(config, sinkhorn_scaling=256), use_fugal=False),
        Algorithm(replace(config, sinkhorn_scaling=512), use_fugal=False),
        Algorithm(replace(config, dynamic_sinkhorn_regularization=True, sinkhorn_regularization=0.5), use_fugal=False)
        #[Algorithm(replace(config, sinkhorn_regularization=0.5, mu=mu), use_fugal=False) for mu in mus],
        #[Algorithm(replace(config, sinkhorn_regularization=0.1, mu=mu), use_fugal=False) for mu in mus],
        #[Algorithm(replace(config, sinkhorn_regularization=0.05, mu=mu), use_fugal=False)for mu in mus],
        #Algorithm(replace(config, sinkhorn_regularization=0.01), use_fugal=False),
        #Algorithm(replace(config, sinkhorn_threshold=1e-1), use_fugal=False),
        #Algorithm(replace(config, sinkhorn_threshold=1e-2), use_fugal=False),
        #Algorithm(replace(config, sinkhorn_threshold=1e-3), use_fugal=False),
        #Algorithm(replace(config, sinkhorn_threshold=1e-4), use_fugal=False),
    ]).flatten(),
    noise_levels=[
        #NoiseLevel(0.0, 0.0, False),
        NoiseLevel(0.1, 0.0, False),
        #NoiseLevel(0.2, 0.0, False),
        #NoiseLevel(0.3, 0.0, False),
    ],
    num_runs=10,
)

#[graph.get(np.random.default_rng()) for graph in experiment.graphs]
results = experiment.run()

print([result.accuracy for _, _, _, result in results.all_results()])
print([sum(result.profile.phase_times.values()) for _, _,_, result in results.all_results()])

folder = "results"
#if not os.path.exists(folder): os.makedirs(folder)
results.dump(folder)