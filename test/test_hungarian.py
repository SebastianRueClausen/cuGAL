import cugal.hungarian_python
import torch
import scipy.optimize
import numpy as np

def test_hungarian():
    for i in range(100):
        T = torch.rand(100, 100)
        #T = torch.tensor([[0.4, 0.2, 0.8], 
        #                  [0.2, 0.3, 0.8], 
        #                  [0.0, 0.8, 0.1]])
        #print(T)
        A = cugal.hungarian_python.hungarian_algorithm(T.clone(), 4)
        #print(T)
        _, B = scipy.optimize.linear_sum_assignment(T, maximize=True)

        #print(' '.join([str(a.item()) for a in A]))
        #print(' '.join([str(b) for b in B]))
        A = [a.item() for a in A]
        eq = np.allclose(A, B)
        if not eq:
            print(' '.join([str(a) for a in A]))
            print(' '.join([str(b) for b in B]))
            print(T.tolist())
            break

if __name__ == '__main__':
    test_hungarian()