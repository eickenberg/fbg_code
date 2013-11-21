import numpy as np

from scipy import sparse
from scipy.sparse import linalg as spl
import time


def _multi_target_ridge(X, Y, M, gamma=1., A=0., alpha=1., 
                        warmstart=None, maxiter=201, rtol=1e-4, verbose=True):

    n, p = X.shape
    n2, T = Y.shape

    assert n2 == n

    if warmstart is None:
        B = np.zeros([p, T])
    else:
        B = warmstart
    
    # if A is None:
    #     A = sparse.eye(Y.shape[1], Y.shape[1])

    # using absolute value on M because we need to make the matrix diag dominant
    d = (np.array(np.abs(M).sum(axis=0)).ravel() + 
         np.array(np.abs(M).sum(axis=1)).ravel()) / 2.

    L = alpha * A + gamma * (sparse.spdiags(d, [0], len(d), len(d)) - M)

    XTX = X.T.dot(X)
    XTY = X.T.dot(Y)
    L_plus_LT = 1. / 2. * (L + L.T)

    def matvec(Bvec):
        if verbose:
            print "matvec"
            t = time.time()
        Bmat = Bvec.reshape(T, p).T
        XTXB = XTX.dot(Bmat)
        BLTL = (L_plus_LT).dot(Bmat.T).T
        if verbose:
            print "matvec took %1.2f seconds" % (time.time() - t)
        
        result = XTXB
        result += BLTL
        return result.ravel('F')

    def rmatvec(Cvec):
        return matvec(Cvec)

    linop = spl.LinearOperator((T * p, T * p), matvec=matvec, rmatvec=rmatvec,
                               dtype=X.dtype)

    linop.rmatvec = rmatvec
    # linop.matvec = matvec

    def f(Bvec):
        # t = time.time()
        # res = np.linalg.norm((X.dot(Bvec.reshape(T, p).T) - Y).ravel(), 2)
        # print "Residual: %1.4f, its calc took %1.2fs" % (res, time.time() - t)
        print "Done one iter"

    # blabla = spl.lsqr(linop, XTY.ravel('F'))
    blabla = spl.cg(linop, XTY.ravel('F'), tol=rtol, 
                    x0=B.ravel('F'), callback=f, maxiter=maxiter)

    sol = blabla[0]

#  
#                        

    return sol.reshape((p, T), order='F')


if __name__ == "__main__":

    rng = np.random.RandomState(42)

    n, p, T = 100, 200, 10

    X = rng.randn(n, p)

    B = rng.randn(p, T - 1)

    B = np.hstack([B, B[:, -1:]])

    noise_level = 3. * np.linalg.norm(X)

    Y = X.dot(B) + rng.randn(n, T) * noise_level

    M = np.eye(T)
    M[-2, -1] = 1
    M[-1, -2] = 1

    warmstart = np.linalg.pinv(X).dot(Y)

    B_est = _multi_target_ridge(X, Y, M, gamma=1000., warmstart=warmstart)

