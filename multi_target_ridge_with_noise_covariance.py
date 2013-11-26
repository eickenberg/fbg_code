import numpy as np
from scipy.sparse.linalg import aslinearoperator, LinearOperator, cg
from sklearn.linear_model.base import LinearModel
from sklearn.decomposition import FactorAnalysis

from scipy.optimize import fmin_l_bfgs_b


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
                           rmatvec=diag_matvec,
                           dtype=np.float64)
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

    return _woodbury_inverse(_diagonal_operator(1. / noise_vars),
                 aslinearoperator(np.linalg.inv(1. / len(activations) * 
                                  activations.T.dot(activations))),
                 components.T, components)


def energy_functional(X, Y, B, invcovB, invcovN, alpha):

    residuals = X.dot(B) - Y
    loss = (residuals.T * invcovN.matvec(residuals.T)).sum()
    pen = (B.T * invcovB.matvec(B.T)).sum()

    return loss + alpha * pen


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
            result = XTXB_invcovN - XTYinvcovN + alpha * B_incovB
            return result.T.ravel()
    else:
        raise(Exception)
        def matvec(vecB):
            XB_minus_Y_invcovN = invcovN.rmatvec(
                (X.dot(vecB.reshape(T, P).T) - Y).T)
            XT_XB_minus_Y_invcovN = X.T.dot(XB_minus_Y_invcovN)
            B_incovB = invcovB.rmatvec(vecB.reshape(T, P)).T
            result = XT_XB_minus_Y_invcovN + alpha * B_incovB
            return result.T.ravel()

    linop = LinearOperator(shape=tuple([X.shape[1] * Y.shape[1]] * 2),
                           matvec=matvec,
                           rmatvec=matvec,
                           dtype=np.dtype('float64'))
    linop.matvec = matvec
    linop.rmatvec = matvec
    linop.dtype = np.dtype('float64')

    return linop



class MultiTaskRidge(LinearModel):

    def __init__(self, invcovB=None, invcovN=None, alpha=1., 
                 cg_callback=None):
        self.invcovB = invcovB
        self.invcovN = invcovN
        self.alpha = alpha
        self.cg_callback = cg_callback

    def fit(self, X, Y):
        self.linop = get_grad_linop(X, Y,
                               self.invcovB, self.invcovN, self.alpha)
        # self.coef_ = cg(self.linop, np.zeros(X.shape[1] * Y.shape[1]),
        #                 callback=self.cg_callback, tol=1e-12)
        def f(vecB):
            B = vecB.reshape(Y.shape[1], X.shape[1]).T
            return energy_functional(X, Y, B, 
                                     self.invcovB, self.invcovN, self.alpha)

        def grad_f(vecB):
            return self.linop.matvec(vecB)

        self.coef_ = fmin_l_bfgs_b(f, np.zeros([Y.shape[1] * X.shape[1]]),
            grad_f, pgtol=1e-12, m=20,iprint=3)[0].reshape(Y.shape[1], 
                                                      X.shape[1])

        return self



if __name__ == "__main__":
    from ridge import _RidgeGridCV

    ridge = _RidgeGridCV(alpha_min=1., alpha_max=1000)

    n_samples, n_features, n_targets = 100, 90, 20

    rng = np.random.RandomState(42)

    # signal_levels = rng.randn(n_targets) ** 2
    signal_levels = np.ones(n_targets)
    global_SNR = 1
    # SNRs = rng.randn(n_targets) ** 2 * global_SNR
    SNRs = np.ones(n_targets) * global_SNR

    beta = rng.randn(n_features, n_targets) * signal_levels

    X_train = rng.randn(n_samples, n_features)
    Y_train_clean = X_train.dot(beta)

    signal_level = Y_train_clean.std(0)

    noise = rng.randn(n_samples, n_targets)
    noise /= noise.std(0)

    Y_train_noisy = Y_train_clean + signal_levels / SNRs * noise

    ridge.fit(X_train, Y_train_noisy)
    ridge_coef = ridge.coef_

    # signal_cov = np.diag(ridge.best_alphas)
    signal_cov = np.diag(np.ones(n_targets))
    noise_cov = np.diag(SNRs)

    signal_inv_cov = aslinearoperator(np.linalg.inv(signal_cov))
    noise_inv_cov = aslinearoperator(np.linalg.inv(noise_cov))

    def cb(*args):
        print "hello"

    mtr = MultiTaskRidge(signal_inv_cov, noise_inv_cov, 1., cg_callback=cb)

    mtr.fit(X_train, Y_train_noisy)
    result = mtr.coef_

    from scipy.optimize import check_grad

    def f(vecB):
        B = vecB.reshape(n_targets, n_features).T
        return energy_functional(X_train,
                                 Y_train_noisy,
                                 B, signal_inv_cov, noise_inv_cov, alpha=1.)

    def grad_f(vecB):
        return mtr.linop.matvec(vecB)

    err = check_grad(f, grad_f, np.ones(n_targets * n_features))



