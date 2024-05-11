import cugal.hungarian_python
import torch
import scipy.optimize
import numpy as np

def test_hungarian():
    T = torch.rand(10000, 10000)
    #print(T)
    A = cugal.hungarian_python.hungarian_algorithm(T, 3)
    _, B = scipy.optimize.linear_sum_assignment(T, maximize=True)

    #print(' '.join([str(a.item()) for a in A]))
    #print(' '.join([str(b) for b in B]))
    A = [a.item() for a in A]
    print(np.allclose(A, B))

if __name__ == '__main__':
    test_hungarian()