from experiment import *
from cugal.config import SinkhornMethod, HungarianMethod, Config
import os
import torch
from dataclasses import replace



config = Config(
    device='cuda:1', 
    sinkhorn_method=SinkhornMethod.LOG,
    dtype=torch.float32,
    sinkhorn_threshold=0,#1e-3,
    sinkhorn_iterations=500,
    mu=2,
    iter_count=15,
    use_sparse_adjacency=True,
    sinkhorn_cache_size=1,
    frank_wolfe_threshold=None,
    frank_wolfe_iter_count=10,
    recompute_distance=True,
    hungarian_method=HungarianMethod.SCIPY,
    sinkhorn_regularization=0.3,
    sinkhorn_regularization=0.3,
    )
experiment = Experiment(
    graphs=[
        #Graph(GraphKind.PREDEFINED_GRAPHS, {
        #    'source_file': 'data/MultiMagna/yeast0_Y2H1.txt',
        #    'target_file': f'data/MultiMagna/yeast{i}_Y2H1.txt',
        #}) for i in range(5, 26, 5)
        #Graph(GraphKind.NEWMAN_WATTS, {'n': 500, 'k': i, 'p': 0.2}) for i in range(2, 10, 2)
        #Graph(GraphKind.NEWMAN_WATTS, {'n': 1000, 'k': 10, 'p': i}) for i in [0.1, 0.2, 0.3, 0.4, 0.5]
        #Graph(GraphKind.NEWMAN_WATTS, {'n': 35000, 'k': 10, 'p': 0.2}),
        Graph(GraphKind.CIT_HEP_PH, {}),
    ],
    algorithms=np.array([
        #Algorithm(config, use_fugal=True),
        #[Algorithm(replace(config, mu=mu, use_fugal=False))                              for mu in mus],
        #[Algorithm(replace(config, sinkhorn_regularization=0.5, mu=mu), use_fugal=False) for mu in mus],
        #[Algorithm(replace(config, sinkhorn_regularization=0.1, mu=mu), use_fugal=False) for mu in mus],
        #[Algorithm(replace(config, sinkhorn_regularization=0.05, mu=mu), use_fugal=False)for mu in mus],
        Algorithm(config, use_fugal=False),
    ]).flatten(),
    noise_levels=[
        NoiseLevel(0.0, 0.0, False),
        #NoiseLevel(0.1, 0.0, False),
        #NoiseLevel(0.2, 0.0, False),
        #NoiseLevel(0.3, 0.0, False),
    ],
    num_runs=1,
    seed=283841238,
)

#[graph.get(np.random.default_rng()) for graph in experiment.graphs]
results = experiment.run()

print([result.accuracy for _, _, _, result in results.all_results()])
print([sum(result.profile.phase_times.values()) for _, _,_, result in results.all_results()])
print([result.profile.sparsity for _, _,_, result in results.all_results()])

folder = "results"
#if not os.path.exists(folder): os.makedirs(folder)
results.dump(folder)
