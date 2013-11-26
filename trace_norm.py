# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

import pylab as plt
import numpy as np

from multitask import trace
from scipy import sparse, linalg
from datetime import datetime
from scipy.sparse import linalg as splinalg

def trace_frankwolfe(X, Y, alpha, rtol=1e-3, max_iter=100, verbose=False,
                     warm_start=None, n_svals=10, L=None, callback=None):
    """
    trace norm constrained Frank-Wolfe. Minimizes a model of the form
    
         0.5 * || Y - X W||_fro ^2

         subject to ||W||_* < alpha
    
    Returns
    -------
    U, S, Vt : factorization of W
    obj_vals : values of the objective function
    time: timing at which the sampling of the obj. function was performed
    """

    n_samples, n_features, n_task = X.shape[0], X.shape[1], Y.shape[1]
    assert X.shape[0] == Y.shape[0]
    U = []
    S = []
    Vt = []
    obj_vals = []
    start = datetime.now()
    time_vals = []
    for i in range(max_iter):
        time_vals.append((datetime.now() - start).total_seconds())
        if len(U):
            res = (Y - X.dot(U).dot(S[:, None] * Vt)) / alpha
            D = S * S
            # We use eigsh instead of svd because we need to use LinearOperator
            matvec = lambda x: np.dot(Vt.T, D * np.dot(Vt, x))
            W = splinalg.LinearOperator(shape=(n_task, n_task), matvec=matvec, dtype=np.float)
            eigvals = splinalg.eigsh(W, k=i)[0]
            svals = np.sqrt(np.abs(eigvals))
        else:
            res = Y / alpha
            svals = np.array(0)

        obj_vals.append(0.5 * (linalg.norm(res, 'fro') ** 2) / alpha + svals.sum())
        if verbose:
            print(obj_vals[-1])
        grad = - X.T.dot(res)

        # conditional gradient step
        u, sv, v = splinalg.svds(grad, k=1, tol=.1, maxiter=10000)
        step = 2. / (i + 2.)
        if len(U):
            U = np.concatenate((U, -u), axis=1)
            Vt = np.concatenate((Vt, v), axis=0)
            S = np.concatenate(((1. - step) * S, step * np.array([1.])))
        else:
            U = -u
            Vt = v
            S = np.array([step])
    return U, S, Vt, obj_vals, time_vals

if __name__ == '__main__':

    n_samples, n_features, n_task = 2000, 1000, 10000
    X = np.random.randn(n_samples, n_features)
    Y = np.random.randn(n_samples, n_task)

    U, S, Vt, obj_vals, time_vals = trace_frankwolfe(X, Y, 10., verbose=False, max_iter=10)
    plt.plot(time_vals, obj_vals)
    plt.ylabel('Objective function')
    plt.xlabel('Time')
    plt.show()

    # <codecell>

    n_samples, n_features, n_task = 500, 1000, 100000
    X = np.random.randn(n_samples, n_features)
    Y = np.random.randn(n_samples, n_task)

    W, obj_vals, time_vals = trace_frankwolfe(X, Y, 1., verbose=False, max_iter=10)
    plt.plot(time_vals, obj_vals)
    plt.ylabel('Objective function')
    plt.xlabel('Time')
    plt.show()

    # <codecell>


