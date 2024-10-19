import torch
from random import uniform
from cugal.config import Config, HungarianMethod
import numpy as np


def greedy_lap(cost_matrix: torch.Tensor, config: Config) -> np.array:
    n = cost_matrix.size(1)
    match config.hungarian_method:
        case HungarianMethod.GREEDY:
            res = list()
            taken = torch.ones(n, device=cost_matrix.device,
                               dtype=cost_matrix.dtype)
            for row in cost_matrix:
                row *= taken
                m = row.argmax()
                taken[m] = 0
                res.append(m.item())
            return res

        case HungarianMethod.RAND:
            order = torch.randperm(n, device=cost_matrix.device)
            taken = torch.ones(n, device=cost_matrix.device,
                               dtype=cost_matrix.dtype)
            res = [0] * n
            for row in order:
                cost_matrix[row] *= taken
                m = cost_matrix[row].argmax()
                taken[m] = 0
                res[row] = m

            return res

        case HungarianMethod.MORE_RAND:
            order = torch.randperm(n, device=cost_matrix.device)
            taken = torch.ones(n, device=cost_matrix.device,
                               dtype=cost_matrix.dtype)
            res = [0] * n
            for row in order:
                cost_matrix[row] *= taken
                cumsum = torch.cumsum(cost_matrix[row], dim=0)
                r = uniform(0, cumsum[-1])
                mask = r <= cumsum
                m = mask.nonzero()[0]
                taken[m] = 0
                res[row] = m.item()

            return res

        case HungarianMethod.DOUBLE_GREEDY:
            max_values_in_rows = cost_matrix.max(dim=1).values
            order = max_values_in_rows.argsort(dim=0, descending=True)

            taken = torch.ones(n, device=cost_matrix.device,
                               dtype=cost_matrix.dtype)
            res = [0] * n
            for row in order:
                cost_matrix[row] *= taken
                m = cost_matrix[row].argmax()
                taken[m] = 0
                res[row] = m.item()

            return res

        case HungarianMethod.PARALLEL_GREEDY:
            max_values_in_rows, max_col_in_rows = cost_matrix.max(dim=1)
            order_values, order = max_values_in_rows.sort(
                dim=0, descending=True)
            half_mask = order_values > 0.5
            parallel_rows = order[half_mask]

            res = torch.zeros(n, device=cost_matrix.device, dtype=torch.long)
            res[parallel_rows] = max_col_in_rows[parallel_rows]

            taken = torch.ones(n, device=cost_matrix.device,
                               dtype=cost_matrix.dtype)
            taken[max_col_in_rows[parallel_rows]] = 0

            for row in order[~half_mask]:
                cost_matrix[row] *= taken
                m = cost_matrix[row].argmax()
                taken[m] = 0
                res[row] = m.item()

            return res
