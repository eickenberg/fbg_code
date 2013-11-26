import numpy as np

from ridge import _RidgeGridCV, _multi_corr_score, _multi_r2_score

n_features = 50
noise_feature_1 = np.zeros(n_features)
noise_feature_1[:int(.6 * n_features)] = 1
noise_feature_2 = np.zeros(n_features)
noise_feature_2[int(.4 * n_features):] = 1

noise_features = np.array([noise_feature_1, noise_feature_2])

n_samples = 200
noise_time_courses = np.random.randn(n_samples, len(noise_features))
noise_time_courses -= noise_time_courses.mean(0)
noise_time_courses /= noise_time_courses.std(0)


feature_variances = np.random.randn(n_features) ** 2

gauss_noise = np.random.randn(n_samples, n_features)
gauss_noise -= gauss_noise.mean(0)
gauss_noise /= gauss_noise.std(0)

gauss_noise *= np.sqrt(feature_variances)


noise = noise_time_courses.dot(noise_features) + gauss_noise

noise_cov = 1. / len(noise) * (noise - noise.mean(0)).T.dot(
    noise -noise.mean(0))

from sklearn.decomposition import FactorAnalysis
fa = FactorAnalysis(n_components=2)

fa.fit(noise)
fcomponents = fa.components_
ftcs = fa.transform(noise)
fvar = fa.noise_variance_

facov = 1. / len(ftcs) * \
    fcomponents.T.dot(ftcs.T.dot(ftcs)).dot(fcomponents) + \
    np.diag(fvar)


n_design_features = 80
design_matrix = np.random.randn(n_samples, n_design_features)

betas = np.random.randn(n_design_features, n_features)

signal = design_matrix.dot(betas)
signal_level = signal.std(0)

SNR = 1.
Y = signal + signal_level / SNR * noise

inv_noise_cov = np.linalg.pinv(noise_cov)
inv_facov = np.linalg.pinv(facov)

from ridge import _RidgeGridCV

train = np.arange(0, int(.8 * n_samples))
test = np.arange(int(.8 * n_samples), n_samples)

ridge = _RidgeGridCV(alpha_min=1., alpha_max=100)

ridge.fit(design_matrix[train], Y[train])
test_pred = ridge.coef_.dot(design_matrix[test].T).T

r2_scores = _multi_r2_score(test_pred, Y[test])
corr_scores = _multi_corr_score(test_pred, Y[test])


from multi_target_ridge_with_noise_covariance import MultiTaskRidge, \
    get_inv_diag_plus_low_rank_cov_op
from scipy.sparse.linalg import aslinearoperator

inv_signal_cov = aslinearoperator(np.diag(ridge.best_alphas))
inv_noise_cov3 = aslinearoperator(inv_noise_cov)
inv_noise_cov2 = get_inv_diag_plus_low_rank_cov_op(noise)
inv_noise_cov = aslinearoperator(inv_facov)



# now do the same test with independently gathered noise
noise_tcs_indep = np.random.randn(n_samples, len(noise_features))
noise_tcs_indep -= noise_tcs_indep.mean(0)
noise_tcs_indep /= noise_tcs_indep.std(0)

gauss_noise_indep = np.random.randn(n_samples, n_features)
gauss_noise_indep -= gauss_noise_indep.mean(0)
gauss_noise_indep /= gauss_noise_indep.std(0)
gauss_noise_indep *= np.sqrt(feature_variances)

noise_indep = noise_tcs_indep.dot(noise_features) + gauss_noise_indep

noise_cov_indep = (noise_indep - noise_indep.mean(0)).T.dot(
    noise_indep - noise_indep.mean(0))

inv_noise_cov_indep = aslinearoperator(np.linalg.pinv(noise_cov_indep))
inv_noise_cov_indep_op = get_inv_diag_plus_low_rank_cov_op(noise_indep,
                                                           rank=2)


mtr = MultiTaskRidge(inv_signal_cov, inv_noise_cov_indep_op, alpha=1.)

mtr.fit(design_matrix[train], Y[train])
test_pred_2 = mtr.coef_.dot(design_matrix[test].T).T
r2_scores_2 = _multi_r2_score(test_pred_2, Y[test])
corr_scores_2 = _multi_corr_score(test_pred_2, Y[test])

import pylab as pl
pl.figure()
pl.plot(corr_scores, corr_scores_2, "r+")
pl.plot([0, 1], [0, 1], "b-")
pl.xlabel("without low rank noise model")
pl.ylabel("with low rank noise model")
pl.title("Comparison low rank noise model")
pl.show()
