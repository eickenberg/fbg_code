"""Simple implementation of a rank k solver for the problem

.5 * ||XUV.T - Y|| ** 2 + .5 * alpha * (||U|| ** 2 + ||V|| ** 2)

"""

import numpy as np
import numexpr as ne
from scipy.optimize import fmin_l_bfgs_b  # XXX better than bfgs
from sklearn.linear_model import Ridge
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt


def f(U, V, X, Y, alpha1, alpha2, out=None, callback=None):
    """The (non-convex) energy functional"""
    # XXX it be nice to have f return func val and grad too

    #loss_value = ((X.dot(U).dot(V.T) - Y) ** 2).sum()
    Z = (X.dot(U).dot(V.T) - Y)
    loss_value = (ne.evaluate("Z * Z")).sum()

    penalization = alpha1 * ne.evaluate("U * U").sum() + alpha2 * ne.evaluate("V * V").sum()

    # Compute gradient
    #XT_residuals = X.T.dot((X.dot(U).dot(V.T) - Y))
    XT_residuals = X.T.dot(Z)

    grad_VT = U.T.dot(XT_residuals) + alpha1 * V.T
    grad_U = XT_residuals.dot(V) + alpha2 * U

    if out is None:
        out = np.empty([U.shape[0] + V.shape[0], U.shape[1]])

    out[:U.shape[0]] = grad_U
    out[U.shape[0]:] = grad_VT.T

    energy = .5 * loss_value + .5 * penalization
    if callback is not None:
        callback(energy)
        
    return energy, out.ravel()

def get_vec_and_grad_func(X, Y, alpha1, alpha2, rank, n_features, callback=None):
    """Returns a function that takes a single column vector and
    reshapes into the apriate rank k matrices and then calculates
    the functional"""

    def vecfunc(vecUV, out=None):
        concat_matrix = vecUV.reshape(-1, rank)
        U = concat_matrix[:n_features]
        V = concat_matrix[n_features:]

        return f(U, V, X, Y, alpha1, alpha2, out=out, callback=callback)

    return vecfunc


def rank_constrained_least_squares(X, Y, rank, alpha1, alpha2=None,
                                   U0=None, V0=None,
                                   max_bfgs_iter=500,
                                   m=10,
                                   gradient_tolerance=1e-5,
                                   callback=None,
                                   verbose=3):
    """
    Minimizes
    .5 * ||XUV.T - Y|| ** 2 + .5 * alpha * (||U|| ** 2 + ||V|| ** 2)

    """

    if alpha2 is None:
        alpha2 = alpha1

    energy_function = get_vec_and_grad_func(X, Y, alpha1, alpha2, rank, X.shape[1], callback=callback)
    #energy_gradient = get_grad_func(X, Y, alpha1, alpha2, rank, len(X.T))

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
    else:
        V = V0.copy()
        if U0 is None:
            raise Exception
        U = U0.copy()

    initial_UV_vec = np.vstack([U, V]).ravel()

    result = fmin_l_bfgs_b(energy_function,
                           x0=initial_UV_vec,
                           #fprime=energy_gradient,
                           #maxiter=max_bfgs_iter,
                           maxfun=max_bfgs_iter,
                           # gtol=gradient_tolerance,
                           m=m,
                           #callback=callback,
                           iprint=verbose)

    concat_matrix = result[0].reshape(-1, rank)
    n_features = X.shape[1]
    U_res = concat_matrix[:n_features]
    V_res = concat_matrix[n_features:]

    return U_res, V_res, result[1:]


if __name__ == "__main__":
    np.random.seed(0)
    
    # test functional and gradient
    n_samples, n_features, n_targets, rank = 40, 50, 30, 4

    X = np.random.randn(n_samples, n_features)
    Y = np.random.randn(n_samples, n_targets)

    B = np.random.randn(n_features, n_targets)
    func = get_vec_and_grad_func(X, Y, 1., 1., rank, n_features)
    #gradient_of_f = get_grad_func(X, Y, 1., 1., rank, n_features)

    from scipy.optimize import check_grad
    for i in range(10):
        U = np.random.randn(n_features, rank)
        V = np.random.randn(n_targets, rank)
        vecUV = np.vstack([U, V]).ravel()
        
        err = check_grad(lambda *args: func(*args)[0],
                         lambda *args: func(*args)[1], vecUV)
        print err

    raise Exception
    maxit = 500
    for r in [5, 6, min(X.shape[1], Y.shape[1]) - 1]:
        print "\r\nCase: r = %i" % r
        energies = []
        func = get_vec_func(X, Y, 1., 1., r, n_features)
        # callback_env = {"it": 0}

        def cb(vec):
            # callback_env["it"] += 1
            energy = func(vec)
            # print "\titer %03i/%03i: energy = %g" % (
            #     callback_env["it"], maxit, energy)
            energies.append(energy)
            # print func(vec)

        result = rank_constrained_least_squares(
            X, Y, r, 1.,
            callback=cb,
            m=10,  # memory budget
            max_bfgs_iter=maxit)
        plt.plot(energies - np.min(energies), label="r=%i" % r)

    # prettify plots
    plt.legend()
    plt.title(("Problem: argmin .5 * ||XUV.T - Y|| ** 2 + .5 * alpha * (||U||"
               " ** 2 + ||V|| ** 2)"))
    plt.ylabel("f(x_k) - f(x*)")
    plt.xlabel("time (s)")
    plt.yscale("log")

    print result

    plt.show()
