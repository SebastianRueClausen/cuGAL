from experiment import *
from cugal.config import SinkhornMethod, HungarianMethod, Config
import os
import torch
from dataclasses import replace

config = Config(
    device='cuda:1', 
    sinkhorn_method=SinkhornMethod.LOG,
    dtype=torch.float32,
    sinkhorn_threshold=0,
    sinkhorn_iterations=500,
    mu=2,
    iter_count=15,
    use_sparse_adjacency=True,
    sinkhorn_cache_size=1,
    frank_wolfe_threshold=0,
    recompute_distance=True,
    hungarian_method=HungarianMethod.SCIPY,
    sinkhorn_regularization=1,
    )

experiment = Experiment(
    graphs=[
        #Graph(GraphKind.PREDEFINED_GRAPHS, {
        #    'source_file': 'data/MultiMagna/yeast0_Y2H1.txt',
        #    'target_file': f'data/MultiMagna/yeast{i}_Y2H1.txt',
        #}) for i in range(5, 26, 5)
        #Graph(GraphKind.NEWMAN_WATTS, {'n': 1000, 'k': 7, 'p': 0.01}),
        #Graph(GraphKind.NEWMAN_WATTS, {'n': 1000, 'k': 7, 'p': 0.05}),
        #Graph(GraphKind.NEWMAN_WATTS, {'n': 1000, 'k': 7, 'p': 0.10}),
        #Graph(GraphKind.NEWMAN_WATTS, {'n': 1000, 'k': 7, 'p': 0.20}),
        Graph(GraphKind.BIO_DMELA, {}),
    ],
    algorithms=[
        Algorithm(config, use_fugal=False),
        Algorithm(replace(config, sinkhorn_regularization=5e-1), use_fugal=False), 
        Algorithm(replace(config, sinkhorn_regularization=1e-1), use_fugal=False),
        Algorithm(replace(config, sinkhorn_regularization=5e-2), use_fugal=False),
        Algorithm(replace(config, sinkhorn_regularization=1e-2), use_fugal=False),
        
        #Algorithm(config, use_fugal=True),
    ],
    noise_levels=[
        NoiseLevel(0.1, 0.0, False),
        NoiseLevel(0.2, 0.0, False),
        NoiseLevel(0.3, 0.0, False),
    ],
)

results = experiment.run()

print([result.accuracy for _, _, _, result in results.all_results()])
print([sum(result.profile.phase_times.values()) for _, _,_, result in results.all_results()])

folder = "results"
if not os.path.exists(folder): os.makedirs(folder)
results.dump(folder)