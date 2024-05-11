import torch
from random import uniform

def hungarian_algorithm(cost_matrix: torch.Tensor, rand):
    """
    Implementation of the Hungarian algorithm using PyTorch.
    
    Parameters:
        cost_matrix (torch.Tensor): The cost matrix for the assignment problem.
    
    Returns:
        list of tuples: A list of tuples where each tuple contains the row and column indices indicating the assignments.
    """
    n = cost_matrix.size(1)
    if rand == 0:
        res = list()
        taken = torch.ones(n, device=cost_matrix.device, dtype=cost_matrix.dtype)
        for row in cost_matrix:
            row *= taken
            m = row.argmax()
            taken[m] = 0
            res.append(m)
        return res

    elif rand == 1:
        order = torch.randperm(n, device=cost_matrix.device)
        taken = torch.ones(n, device=cost_matrix.device, dtype=cost_matrix.dtype)
        res = [0] * n
        for row in order:
            cost_matrix[row] *= taken
            m = cost_matrix[row].argmax()
            taken[m] = 0
            res[row] = m

        return res
    
    elif rand == 2:
        order = torch.randperm(n, device=cost_matrix.device)
        taken = torch.ones(n, device=cost_matrix.device, dtype=cost_matrix.dtype)
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
    
    elif rand == 3:
        max_values = cost_matrix.max(dim=1).values
        #[print(", ".join([f"{v:.2f}" for v in V])) for V in cost_matrix.tolist()]
        order = max_values.argsort(dim=0, descending=True)

        taken = torch.ones(n, device=cost_matrix.device, dtype=cost_matrix.dtype)
        res = [0] * n
        for row in order:
            cost_matrix[row] *= taken
            m = cost_matrix[row].argmax()
            taken[m] = 0
            res[row] = m

        return res
    else:
        raise ValueError("Invalid value for rand. Must be 0, 1, 2 or 3.")


    num_agents, num_tasks = cost_matrix.size()
    
    # Step 1: Subtract the minimum value of each row from all elements of that row
    cost_matrix -= torch.min(cost_matrix, dim=1, keepdim=True)[0]
    
    # Step 2: Subtract the minimum value of each column from all elements of that column
    cost_matrix -= torch.min(cost_matrix, dim=0, keepdim=True)[0]
    
    # Step 3: Cover all zeros in the cost matrix using minimum number of lines
    covered_zeros = torch.zeros_like(cost_matrix, dtype=torch.uint8)
    row_covered = torch.zeros(num_agents, dtype=torch.uint8)
    col_covered = torch.zeros(num_tasks, dtype=torch.uint8)
    
    while covered_zeros.sum() < min(num_agents, num_tasks):
        # Find the first zero not already covered
        i, j = ((1 - row_covered.unsqueeze(1)) * (1 - col_covered)).nonzero()[0]
        
        # Cover the zero
        covered_zeros[i, j] = 1
        
        # Find other zeros in the same row and column and cover them
        row_covered[i] = 1
        col_covered[j] = 1
        
    # Step 4: Determine the minimum number of lines to cover all zeros
    num_covered_rows = row_covered.sum()
    
    # If all zeros are covered, go to Step 7
    if num_covered_rows == min(num_agents, num_tasks):
        assignments = []
        for i in range(num_agents):
            for j in range(num_tasks):
                if covered_zeros[i, j] == 1:
                    assignments.append((i, j))
        return assignments
    
    # Step 5: Find the smallest uncovered entry and subtract it from all uncovered entries
    min_uncovered_entry = cost_matrix[(1 - row_covered).unsqueeze(1) * (1 - col_covered)].min()
    cost_matrix -= min_uncovered_entry
    
    # Step 6: Find the smallest entry not covered by any line and add it to all entries covered by two lines
    uncovered_entries = (1 - row_covered).unsqueeze(1) * (1 - col_covered)
    min_uncovered_entry = cost_matrix[uncovered_entries].min()
    cost_matrix[uncovered_entries] += min_uncovered_entry
    
    # Repeat from Step 3
    return hungarian_algorithm(cost_matrix)
