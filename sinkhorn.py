# -*- coding: utf-8 -*-
"""
Rewrite ot.bregman.sinkhorn in Python Optimal Transport (https://pythonot.github.io/_modules/ot/bregman.html#sinkhorn)
using pytorch operations.
Bregman projections for regularized OT (Sinkhorn distance).
"""

import numpy as np
import time

M_EPS = 1e-16


def sinkhorn(a, b, C, reg=1e-1, method='sinkhorn', maxIter=1000, tau=1e3,
             stopThr=1e-9, verbose=False, log=False, warm_start=None, eval_freq=10, print_freq=200, **kwargs):
    """
    Solve the entropic regularization optimal transport
    The input should be PyTorch tensors
    The function solves the following optimization problem:

    .. math::
        \gamma = arg\min_\gamma <\gamma,C>_F + reg\cdot\Omega(\gamma)
        s.t. \gamma 1 = a
             \gamma^T 1= b
             \gamma\geq 0
    where :
    - C is the (ns,nt) metric cost matrix
    - :math:`\Omega` is the entropic regularization term :math:`\Omega(\gamma)=\sum_{i,j} \gamma_{i,j}\log(\gamma_{i,j})`
    - a and b are target and source measures (sum to 1)
    The algorithm used for solving the problem is the Sinkhorn-Knopp matrix scaling algorithm as proposed in [1].

    Parameters
    ----------
    a : torch.tensor (na,)
        samples measure in the target domain
    b : torch.tensor (nb,)
        samples in the source domain
    C : torch.tensor (na,nb)
        loss matrix
    reg : float
        Regularization term > 0
    method : str
        method used for the solver either 'sinkhorn', 'greenkhorn', 'sinkhorn_stabilized' or
        'sinkhorn_epsilon_scaling', see those function for specific parameters
    maxIter : int, optional
        Max number of iterations
    stopThr : float, optional
        Stop threshol on error ( > 0 )
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True

    Returns
    -------
    gamma : (na x nb) torch.tensor
        Optimal transportation matrix for the given parameters
    log : dict
        log dictionary return only if log==True in parameters

    References
    ----------
    [1] M. Cuturi, Sinkhorn Distances : Lightspeed Computation of Optimal Transport, Advances in Neural Information Processing Systems (NIPS) 26, 2013
    See Also
    --------

    """

    if method.lower() == 'sinkhorn':
        return sinkhorn_knopp(a, b, C, reg, maxIter=maxIter,
                              stopThr=stopThr, verbose=verbose, log=log,
                              warm_start=warm_start, eval_freq=eval_freq, print_freq=print_freq,
                              **kwargs)
    else:
        raise ValueError("Unknown method '%s'." % method)


def sinkhorn_knopp(a, b, C, reg=1e-1, maxIter=1000, stopThr=1e-9,
                   verbose=False, log=False, warm_start=None, eval_freq=10, print_freq=200, **kwargs):
    """
    Solve the entropic regularization optimal transport
    The input should be PyTorch tensors
    The function solves the following optimization problem:

    .. math::
        \gamma = arg\min_\gamma <\gamma,C>_F + reg\cdot\Omega(\gamma)
        s.t. \gamma 1 = a
             \gamma^T 1= b
             \gamma\geq 0
    where :
    - C is the (ns,nt) metric cost matrix
    - :math:`\Omega` is the entropic regularization term :math:`\Omega(\gamma)=\sum_{i,j} \gamma_{i,j}\log(\gamma_{i,j})`
    - a and b are target and source measures (sum to 1)
    The algorithm used for solving the problem is the Sinkhorn-Knopp matrix scaling algorithm as proposed in [1].

    Parameters
    ----------
    a : torch.tensor (na,)
        samples measure in the target domain
    b : torch.tensor (nb,)
        samples in the source domain
    C : torch.tensor (na,nb)
        loss matrix
    reg : float
        Regularization term > 0
    maxIter : int, optional
        Max number of iterations
    stopThr : float, optional
        Stop threshol on error ( > 0 )
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True

    Returns
    -------
    gamma : (na x nb) torch.tensor
        Optimal transportation matrix for the given parameters
    log : dict
        log dictionary return only if log==True in parameters

    References
    ----------
    [1] M. Cuturi, Sinkhorn Distances : Lightspeed Computation of Optimal Transport, Advances in Neural Information Processing Systems (NIPS) 26, 2013
    See Also
    --------

    """

    na, nb = C.shape

    assert na >= 1 and nb >= 1, 'C needs to be 2d'
    assert na == a.shape[0] and nb == b.shape[0], "Shape of a or b does't match that of C"
    assert reg > 0, 'reg should be greater than 0'
    assert a.min() >= 0. and b.min() >= 0., 'Elements in a or b less than 0'

    if log:
        log = {'err': []}

    if warm_start is not None:
        u = warm_start['u']
        v = warm_start['v']
    else:
        u = np.ones(na) / na
        v = np.ones(nb) / nb

    K = np.empty(shape=C.shape)
    np.divide(C, -reg, out=K)
    np.exp(K, out=K)

    b_hat = np.empty(shape=b.shape)

    it = 1
    err = 1

    # allocate memory beforehand
    KTu = np.empty(shape=v.shape)
    Kv = np.empty(shape=u.shape)

    # t1 = time.time()
    while (err > stopThr and it <= maxIter):
        upre, vpre = u, v
        np.matmul(u, K, out=KTu)
        v = np.divide(b, KTu + M_EPS)
        np.matmul(K, v, out=Kv)
        u = np.divide(a, Kv + M_EPS)

        if np.any(np.isnan(u)) or np.any(np.isnan(v)) or \
                np.any(np.isinf(u)) or np.any(np.isinf(v)):
            print('Warning: numerical errors at iteration', it)
            u, v = upre, vpre
            break

        if log and it % eval_freq == 0:
            # we can speed up the process by checking for the error only all
            # the eval_freq iterations
            # below is equivalent to:
            # b_hat = torch.sum(u.reshape(-1, 1) * K * v.reshape(1, -1), 0)
            # but with more memory efficient
            b_hat = np.matmul(u, K) * v
            err = (b - b_hat).pow(2).sum().item()
            # err = (b - b_hat).abs().sum().item()
            log['err'].append(err)

        if verbose and it % print_freq == 0:
            print('iteration {:5d}, constraint error {:5e}'.format(it, err))

        it += 1
    # t2 = time.time()
    # print("Sinkhorn loop: ", t2 - t1)

    if log:
        log['u'] = u
        log['v'] = v
        log['alpha'] = reg * np.log(u + M_EPS)
        log['beta'] = reg * np.log(v + M_EPS)

    # transport plan
    P = u.reshape(-1, 1) * K * v.reshape(1, -1)
    if log:
        return P, log
    else:
        return P
