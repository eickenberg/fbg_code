"""Simple implementation of a rank k solver for the problem

.5 * ||XUV.T - Y|| ** 2 + .5 * alpha * (||U|| ** 2 + ||V|| ** 2)

"""

import numpy as np


def f(U, V, X, Y, alpha):
    """The (non-convex) energy functional"""
    loss_value = ((X.dot(U).dot(V.T) - Y) ** 2).sum()

    penalization = (U ** 2).sum() + (V ** 2).sum()

    return loss_value + alpha * penalization


def grad_f(U, V, X, Y, alpha, out=None):
    """The gradient of the energy functional in np.vstack([U, V])"""
    XT_residuals = X.T.dot((X.dot(U).dot(V.T) - Y))

    grad_VT = U.T.dot(XT_residuals)
    grad_U = XT_residuals.dot(V)

    if out is None:
        out = np.empty([U.shape[0] + V.shape[0], U.shape[1]])

    out[:U.shape[0]] = grad_U
    out[U.shape[0]:] = grad_VT.T

    return out


def get_vec_func(X, Y, alpha, rank, n_samples):
    """Returns a function that takes a single column vector and
    reshapes into the appropriate rank k matrices and then calculates
    the functional"""

    def vecfunc(vecUV):
        concat_matrix = vecUV.reshape(-1, rank)
        U = concat_matrix[:n_samples]
        V = concat_matrix[n_samples:]

        return f(U, V, X, Y, alpha)

    return vecfunc


def get_grad_func(X, Y, alpha, rank, n_samples):
    """Returns function that takes a single column vector and
    reshapes into the appropriate rank k matrices and then calculates
    the gradient of the functional"""

    def vecgrad(vecUV, out=None):
        concat_matrix = vecUV.reshape(-1, rank)
        U = concat_matrix[:n_samples]
        V = concat_matrix[n_samples:]
