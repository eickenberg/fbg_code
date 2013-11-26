import numpy as np
import data
import cortex
import scipy.sparse
#from rank_k_bfgs import rank_constrained_least_squares, get_vec_and_grad_func
import rank_k_bfgs
import matplotlib.pyplot as plt

from ridge import _multi_corr_score, _multi_r2_score


numtime = 1000

# Load stimuli
trnstim = data.get_wordnet("train")[:numtime]
valstim = data.get_wordnet("val")

delays = [2, 3, 4]
deltrnstim = np.hstack([np.roll(trnstim, d, 0) for d in delays])
delvalstim = np.hstack([np.roll(valstim, d, 0) for d in delays])

#sdeltrnstim = scipy.sparse.csr_matrix(deltrnstim)
#sdelvalstim = scipy.sparse.csr_matrix(delvalstim)

zs = lambda m: (m - m.mean(0)) / m.std(0)

sdeltrnstim = deltrnstim = np.nan_to_num(zs(deltrnstim))
sdelvalstim = delvalstim = np.nan_to_num(zs(delvalstim))

# Select some voxels
ebamask = cortex.get_roi_mask("MLfs", "20121210ML_auto1", roi="EBA")["EBA"] > 0

# Load training, test fMRI data
trndata = data.get_train(masked=ebamask)[:numtime]
valdata = data.get_val(masked=ebamask)

from ridge import _RidgeGridCV

ridge = _RidgeGridCV(alpha_min=1., alpha_max=1000., n_grid_points=5,
                     n_grid_refinements=2, cv=2)

ridge_coefs = ridge.fit(deltrnstim, trndata).coef_.T
Uridge, sridge, VridgeT = np.linalg.svd(ridge_coefs, full_matrices=False)

ranks = [1, 2, 5, 10]

results = []
corr_scores = []
r2_scores = []
ridge_corr_scores = []
ridge_r2_scores = []

for r in ranks:
    print "Rank: %d" % r
    U_, V_, something = result = \
        rank_k_bfgs.rank_constrained_least_squares(sdeltrnstim, trndata, r,
                                                        0.1, 0.1,
                   # U0=rng.randn(X_train.shape[1], r),
                   # V0=rng.randn(Y_train_noisy.shape[1], r),
                     U0=Uridge[:, :r] * np.sqrt(sridge[:r]),
                     V0=VridgeT[:r].T * np.sqrt(sridge[:r]),
                     max_bfgs_iter=2000
                                                   )
    results.append(result)

    predictions = sdelvalstim.dot(U_).dot(V_.T)
    corr_scores.append(_multi_corr_score(valdata, predictions))
    r2_scores.append(_multi_r2_score(valdata, predictions))

    rank_k_ridge_coefs = Uridge[:, :r].dot(
        sridge[:r][:, np.newaxis] * VridgeT[:r, :])

    ridge_predictions = sdelvalstim.dot(rank_k_ridge_coefs)
    ridge_corr_scores.append(_multi_corr_score(valdata,
                                               ridge_predictions))
    ridge_r2_scores.append(_multi_r2_score(valdata, ridge_predictions))
