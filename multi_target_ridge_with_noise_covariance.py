import numpy as np
from scipy.sparse.linalg import aslinearoperator, LinearOperator, cg
from sklearn.linear_model.base import LinearModel
from sklearn.decomposition import FactorAnalysis



def _diagonal_operator(diag):
    """Creates an operator representing a 
    multiplication with a diagonal matrix"""
    diag = diag.ravel()[:, np.newaxis]

    def diag_matvec(vec):
        if vec.ndim > 1:
            return diag * vec
        else:
            return diag.ravel() * vec

    linop = LinearOperator(shape=(len(diag), len(diag)),
                           matvec=diag_matvec,
                           rmatvec=diag_matvec)
    linop.matvec = diag_matvec
    linop.rmatvec = diag_matvec

    return linop


def test_diagonal_operator():
    diag1 = np.arange(5)
    diagop1 = _diagonal_operator(diag1)

    vec1 = np.ones([len(diag1), 1])
    assert (diagop1.matvec(vec1) == diag1[:, np.newaxis] * vec1).all()
    assert (diagop1.matvec(vec1.ravel()) == diag1).all()

    vec2 = np.hstack([vec1] * 3)
    assert (diagop1.matvec(vec2) == diag1[:, np.newaxis] * vec2).all()



def _woodbury_inverse(Ainv, Cinv, U, V):
    """Uses Woodbury Matrix Identity to invert the Matrix
    (A + UCV) ^ (-1)
    See http://en.wikipedia.org/wiki/Woodbury_matrix_identity"""

    def matvec(x):
        # this is probably wildly suboptimal, but it works
        Ainv_x = Ainv.matvec(x)
        Cinv_mat = Cinv.matvec(np.eye(Cinv.shape[0]))
        VAinvU = V.dot(Ainv.matvec(U))
        inv_Cinv_plus_VAinvU = np.linalg.inv(Cinv_mat + VAinvU)
        VAinv_x = V.dot(Ainv_x)
        inv_blabla_VAinv_x = inv_Cinv_plus_VAinvU.dot(VAinv_x)
        whole_big_block = Ainv.matvec(
            U.dot(inv_blabla_VAinv_x))
        return Ainv_x - whole_big_block

    shape = Ainv.shape
    linop = LinearOperator(shape=shape, matvec=matvec)
    linop.matvec = matvec
    linop.rmatvec = matvec
    return linop


def test_woodbury_inverse():
    rng = np.random.RandomState(42)
    p = 100
    r = 10

    diag = rng.randn(p) ** 2

    lr = rng.randn(r, p)

    m = np.diag(diag) + lr.T.dot(lr)

    inv_m = np.linalg.inv(m)

    woodbury_inv = _woodbury_inverse(_diagonal_operator(1. / diag),
                                     _diagonal_operator(np.ones(r)),
                                     lr.T, lr)

    assert ((inv_m - woodbury_inv.matvec(np.eye(p))) ** 2).sum() < 1e-5


def get_inv_diag_plus_low_rank_cov_op(X, rank=2):
    fa = FactorAnalysis(n_components=rank)
    fa.fit(X)
    components = fa.components_
    noise_vars = fa.noise_variance_
    activations = fa.transform(X)

    return _woodbury_inverse(_diagonal_operator(noise_vars),
                 aslinearoperator(activations.T.dot(activations)),
                 components.T, components)


def get_grad_linop(X, Y, invcovB, invcovN, alpha):
    """
    Linear operator implementing the gradient of the functional
    \frac{1}{2} \|Y - XB\|^2_{\Sigma_n} + \frac{1}{2} \|B\|^2_{\Sigma_s}

    which reads
    grad_B = X^T(XB - Y)\Sigma_n^{-1} + \lambda B\Sigma_s^{-1}
    """

    N, P = X.shape
    T = invcovB.shape[0]

    if P <= N:
        XTX = aslinearoperator(X.T.dot(X))
        XTYinvcovN = invcovN.rmatvec(Y.T.dot(X)).T

        def matvec(vecB):
            XTXB = XTX.matvec(vecB.reshape(T, P).T)
            XTXB_invcovN = invcovN.rmatvec(XTXB.T).T
            B_incovB = invcovB.rmatvec(vecB.reshape(T, P)).T
            return XTXB - XTYinvcovN + alpha * B_incovB
    else:
        def matvec(vecB):
            XB_minus_Y_invcovN = invcovN.rmatvec(
                (X.dot(vecB.reshape(T, P).T) - Y).T)
            XT_XB_minus_Y_invcovN = X.T.dot(XB_minus_Y_invcovN)
            B_incovB = invcovB.rmatvec(vecB.reshape(T, P)).T
            return XT_XB_minus_Y_invcovN + alpha * B_incovB

    linop = LinearOperator(shape=tuple([X.shape[1] * Y.shape[1]] * 2),
                           matvec=matvec,
                           rmatvec=matvec)
    linop.matvec = matvec
    linop.rmatvec = matvec

    return linop



class MultiTaskRidge(LinearModel):

    def __init__(self, invcovB=None, invcovN=None, alpha=1.):
        self.invcovB = invcovB
        self.invcovN = invcovN
        self.alpha = alpha

    def fit(self, X, Y):
        linop = get_grad_linop(X, Y, 
                               self.invcovB, self.invcovN, self.alpha)
        self.coef_ = cg(linop, np.zeros(X.shape[1] * Y.shape[1]))

        return self



if __name__ == "__main__":
    test_woodbury_inverse()
    
