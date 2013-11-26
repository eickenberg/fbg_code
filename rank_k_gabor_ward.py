import numpy as np
import data
import cortex
import scipy.sparse
from sklearn.cluster import WardAgglomeration
from sklearn.feature_extraction import image
import rank_k_bfgs
import pylab as pl

from ridge import _multi_corr_score, _multi_r2_score

numtime = 1000
numclusters = 100

# Load stimuli
trnstim = data.get_gabor("train").T
valstim = data.get_gabor("val").T

delays = [2, 3, 4]
deltrnstim = np.hstack([np.roll(trnstim, d, 0) for d in delays])[:numtime]
delvalstim = np.hstack([np.roll(valstim, d, 0) for d in delays])

#sdeltrnstim = scipy.sparse.csr_matrix(deltrnstim)
#sdelvalstim = scipy.sparse.csr_matrix(delvalstim)

zs = lambda m: (m - m.mean(0)) / m.std(0)

sdeltrnstim = deltrnstim = np.nan_to_num(zs(deltrnstim))
sdelvalstim = delvalstim = np.nan_to_num(zs(delvalstim))

# Select some voxels
cort_mask = cortex.get_cortical_mask("MLfs", "20121210ML_auto1", "thick")
#rois = ["V1", "V2", "V3"]
rois = ["V1"]
masks = [cortex.get_roi_mask("MLfs",
                             "20121210ML_auto1",
                             roi=roi)[roi] > 0 for roi in rois]
roimask = reduce(lambda x, y: (x + y), masks)
wardmask = cort_mask - roimask

# Load training, test fMRI data
trndata_roi = np.nan_to_num(data.get_train(masked=roimask)[:numtime])
trndata_ward = np.nan_to_num(data.get_train(masked=wardmask)[:numtime])

connectivity = image.grid_to_graph(n_x=wardmask.shape[0],
                                   n_y=wardmask.shape[1],
                                   n_z=wardmask.shape[2],
    mask=wardmask)
ward = WardAgglomeration(n_clusters=numclusters, connectivity=connectivity,
                         memory='nilearn_cache')
ward.fit(trndata_ward)
labels = ward.labels_
trndata_collapsed = np.array([trndata_ward[:, labels == i].mean(1)
                              for i in range(numclusters)])
trndata = np.hstack((trndata_roi, trndata_collapsed.T))
valdata = data.get_val(masked=roimask)

from ridge import _RidgeGridCV

ridge = _RidgeGridCV(alpha_min=1., alpha_max=1000., n_grid_points=5,
                     n_grid_refinements=2, cv=2)

ridge_coefs = ridge.fit(sdeltrnstim, trndata).coef_.T
Uridge, sridge, VridgeT = np.linalg.svd(ridge_coefs, full_matrices=False)

ranks = [1, 20, 50, 100, 500]

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
    corr_scores.append(_multi_corr_score(valdata,
                                         predictions[:, :trndata_roi.shape[1]]))
    r2_scores.append(_multi_r2_score(valdata,
                                    predictions[:, :trndata_roi.shape[1]]))

    rank_k_ridge_coefs = Uridge[:, :r].dot(
        sridge[:r][:, np.newaxis] * VridgeT[:r, :])

    ridge_predictions = sdelvalstim.dot(rank_k_ridge_coefs)
    ridge_corr_scores.append(_multi_corr_score(valdata,
                                               ridge_predictions[:, :trndata_roi.shape[1]]))
    ridge_r2_scores.append(_multi_r2_score(valdata,
                                           ridge_predictions[:, :trndata_roi.shape[1]]))


n_targets = trndata.shape[1]
pl.figure(1)
pl.clf()
corr_array = np.nan_to_num(np.array(corr_scores))
# pl.plot(ranks, corr_array, 'k-', lw=0.5)
pl.errorbar(ranks,
            corr_array.mean(axis=1),
            yerr=corr_array.std(axis=1) / np.sqrt(n_targets),
            elinewidth=2,
            linewidth=2,
            color="r")
ridge_corr_array = np.nan_to_num(np.array(ridge_corr_scores))
pl.errorbar(ranks,
            ridge_corr_array.mean(axis=1),
            yerr=ridge_corr_array.std(axis=1) / np.sqrt(n_targets),
            elinewidth=2,
            linewidth=2,
            color="b")
max_corr = max(corr_array.mean(1).max(), ridge_corr_array.mean(1).max())
pl.axis([0, max(ranks), -.1, max_corr + 0.1])
pl.title("Correlation scores")

pl.figure(2)
pl.clf()
# pl.plot(np.array(r2_scores), 'k-', lw=0.5)
r2_array = np.nan_to_num(np.array(r2_scores))
pl.errorbar(ranks,
            r2_array.mean(axis=1),
            yerr=r2_array.std(axis=1) / np.sqrt(n_targets),
            elinewidth=2,
            linewidth=2,
            color="r")
ridge_r2_array = np.nan_to_num(np.array(ridge_r2_scores))
pl.errorbar(ranks,
            ridge_r2_array.mean(axis=1),
            yerr=ridge_r2_array.std(axis=1) / np.sqrt(n_targets),
            elinewidth=2,
            linewidth=2,
            color="b")
pl.axis([0, max(ranks), -1.1, 1.1])
pl.title("R2 scores")
