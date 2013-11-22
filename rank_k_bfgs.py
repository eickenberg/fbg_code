"""Simple implementation of a rank k solver for the problem

.5 * ||XUV.T - Y|| ** 2 + .5 * alpha * (||U|| ** 2 + ||V|| ** 2)

"""

import numpy as np
from scipy.optimize import fmin_bfgs
from sklearn.linear_model import Ridge
from scipy.sparse.linalg import svds


def f(U, V, X, Y, alpha1, alpha2):
    """The (non-convex) energy functional"""
    loss_value = ((X.dot(U).dot(V.T) - Y) ** 2).sum()

    penalization = alpha1 * (U ** 2).sum() + alpha2 * (V ** 2).sum()

    return .5 * loss_value + .5 * penalization


def grad_f(U, V, X, Y, alpha1, alpha2, out=None):
    """The gradient of the energy functional in np.vstack([U, V])"""
    XT_residuals = X.T.dot((X.dot(U).dot(V.T) - Y))

    grad_VT = U.T.dot(XT_residuals) + alpha1 * V.T
    grad_U = XT_residuals.dot(V) + alpha2 * U

    if out is None:
        out = np.empty([U.shape[0] + V.shape[0], U.shape[1]])

    out[:U.shape[0]] = grad_U
    out[U.shape[0]:] = grad_VT.T

    return out


def get_vec_func(X, Y, alpha1, alpha2, rank, n_features):
    """Returns a function that takes a single column vector and
    reshapes into the appropriate rank k matrices and then calculates
    the functional"""

    def vecfunc(vecUV):
        concat_matrix = vecUV.reshape(-1, rank)
        U = concat_matrix[:n_features]
        V = concat_matrix[n_features:]

        return f(U, V, X, Y, alpha1, alpha2)

    return vecfunc


def get_grad_func(X, Y, alpha1, alpha2, rank, n_features):
    """Returns function that takes a single column vector and
    reshapes into the appropriate rank k matrices and then calculates
    the gradient of the functional"""

    def vecgrad(vecUV, out=None):
        concat_matrix = vecUV.reshape(-1, rank)
        U = concat_matrix[:n_features]
        V = concat_matrix[n_features:]

        gradient = grad_f(U, V, X, Y, alpha1, alpha2, out=out)

        return gradient.ravel()

    return vecgrad


def rank_constrained_least_squares(X, Y, rank, alpha1, alpha2=None, 
                                   U0=None, V0=None, 
                                   max_bfgs_iter=500,
                                   gradient_tolerance=1e-5,
                                   callback=None):
    """Minimizes 

    .5 * ||XUV.T - Y|| ** 2 + .5 * alpha * (||U|| ** 2 + ||V|| ** 2)

    """
    if alpha2 is None:
        alpha2 = alpha1

    energy_function = get_vec_func(X, Y, alpha1, alpha2, rank, len(X.T))
    energy_gradient = get_grad_func(X, Y, alpha1, alpha2, rank, len(X.T))

    # if not already done, initialize U and V
    if V0 is None:
        if U0 is not None:
            # if only V0 is None initialize U with a least squares
            U = U0.copy()
            V = np.linalg.pinv(X.dot(U)).dot(Y)
        else:
            # decompose a ridge solution
            _, largest_singular_value_of_X, _ = svds(X, k=1)
            ridge_penalty = largest_singular_value_of_X * .1
            ridge = Ridge(alpha=ridge_penalty)
            ridge_coef = ridge.fit(X, Y).coef_.T
            U, s, VT = svds(ridge_coef, k=rank)
            V = VT.T * np.sqrt(s)
            U *= np.sqrt(s)[np.newaxis, :]

    initial_UV_vec = np.vstack([U, V]).ravel()

    result = fmin_bfgs(energy_function, x0=initial_UV_vec, 
                       fprime=energy_gradient, maxiter=max_bfgs_iter,
                       gtol=gradient_tolerance,
                       callback=callback)

    concat_matrix = result.reshape(-1, rank)
    U_res = concat_matrix[:n_features]
    V_res = concat_matrix[n_features:]

    return U_res, V_res


if __name__ == "__main__":
    # test functional and gradient
    n_samples, n_features, n_targets, rank = 40, 50, 30, 4

    X = np.random.randn(n_samples, n_features)
    Y = np.random.randn(n_samples, n_targets)

    # B = np.random.randn(n_features, n_targets)
    # func = get_vec_func(X, Y, 1., 1., rank, n_features)
    # gradient_of_f = get_grad_func(X, Y, 1., 1., rank, n_features)

    # from scipy.optimize import check_grad
    # for i in range(10):
    #     U = np.random.randn(n_features, rank)
    #     V = np.random.randn(n_targets, rank)
    #     vecUV = np.vstack([U, V]).ravel()

    #     err = check_grad(func, gradient_of_f, vecUV)
    #     print err


    energies = []
    for r in [5, 6, min(X.shape[1], Y.shape[1]) - 1]:
        func = get_vec_func(X, Y, 1., 1., r, n_features)
        def cb(vec):
            print func(vec)

        result = rank_constrained_least_squares(X, Y, r, 1., callback=cb)

    print result
