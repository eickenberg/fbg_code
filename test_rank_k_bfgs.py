import numpy as np

import rank_k_bfgs
from ridge import _multi_corr_score, _multi_r2_score

# Generate some data according to a noisy linear model

n_samples, n_features, n_targets, rank = 200, 500, 150, 10

SNR = 0.5

rng = np.random.RandomState(42)

# let's make it a Gaussian X first, but possibly we need to imitate
# the structure of our design matrix to perform a real test
X_train = rng.randn(n_samples, n_features)
U = rng.randn(n_features, rank)
V = rng.randn(n_targets, rank)

Y_train_clean = X_train.dot(U).dot(V.T)
signal_stdev = Y_train_clean.std(axis=0)
noise_vector = rng.randn(*Y_train_clean.shape)
noise_vector /= noise_vector.std(axis=0)

Y_train_noisy = Y_train_clean + signal_stdev / SNR * noise_vector



num_test_samples = 100
test_SNR = 100.
X_test = rng.randn(num_test_samples, n_features)
Y_test_clean = X_test.dot(U).dot(V.T)
Y_test_stdev = Y_test_clean.std(0)
test_noise = rng.randn(*Y_test_clean.shape)
test_noise /= test_noise.std(axis=0)

Y_test_noisy = Y_test_clean + Y_test_stdev / test_SNR * test_noise


from ridge import _RidgeGridCV

ridge = _RidgeGridCV(alpha_min=1., alpha_max=1000., n_grid_points=5,
                     n_grid_refinements=2, cv=2)

ridge_coefs = ridge.fit(X_train, Y_train_noisy).coef_.T

Uridge, sridge, VridgeT = np.linalg.svd(ridge_coefs, full_matrices=False)

max_rank = min(n_targets, n_features)
#ranks = np.arange(max_rank) + 1
ranks = np.hstack([np.arange(1,15), np.unique(np.linspace(20, max_rank-1, 10).astype(int))])
results = []
corr_scores = []
r2_scores = []
ridge_corr_scores = []
ridge_r2_scores = []

for r in ranks:
    print "Rank: %d" % r
    U_, V_, something = result = \
        rank_k_bfgs.rank_constrained_least_squares(X_train, Y_train_noisy, r,
                                                        0.1, 0.1,
                   # U0=rng.randn(X_train.shape[1], r),
                   # V0=rng.randn(Y_train_noisy.shape[1], r),
                     U0=Uridge[:, :r] * np.sqrt(sridge[:r]),
                     V0=VridgeT[:r].T * np.sqrt(sridge[:r]),
                     max_bfgs_iter=1000
                                                   )
    results.append(result)

    predictions = X_test.dot(U_).dot(V_.T)
    corr_scores.append(_multi_corr_score(Y_test_noisy, predictions))
    r2_scores.append(_multi_r2_score(Y_test_noisy, predictions))

    rank_k_ridge_coefs = Uridge[:, :r].dot(
        sridge[:r][:, np.newaxis] * VridgeT[:r, :])

    ridge_predictions = X_test.dot(rank_k_ridge_coefs)
    ridge_corr_scores.append(_multi_corr_score(Y_test_noisy,
                                               ridge_predictions))
    ridge_r2_scores.append(_multi_r2_score(Y_test_noisy, ridge_predictions))


import pylab as pl
n_targets = trndata.shape[1]
pl.figure(1)
pl.clf()
corr_array = np.array(corr_scores)
# pl.plot(ranks, corr_array, 'k-', lw=0.5)
pl.errorbar(ranks, 
            corr_array.mean(axis=1), 
            yerr=corr_array.std(axis=1) / np.sqrt(n_targets),
            elinewidth=2,
            linewidth=2,
            color="r")
ridge_corr_array = np.array(ridge_corr_scores)
pl.errorbar(ranks, 
            ridge_corr_array.mean(axis=1),
            yerr=ridge_corr_array.std(axis=1) / np.sqrt(n_targets),
            elinewidth=2,
            linewidth=2,
            color="b")
max_corr = max(corr_array.mean(1).max(), ridge_corr_array.mean(1).max())
pl.axis([0, max_rank, -.1, max_corr + 0.1])
pl.title("Correlation scores")

pl.figure(2)
pl.clf()
# pl.plot(np.array(r2_scores), 'k-', lw=0.5)
r2_array = np.array(r2_scores)
pl.errorbar(ranks, 
            r2_array.mean(axis=1),
            yerr=r2_array.std(axis=1) / np.sqrt(n_targets),
            elinewidth=2,
            linewidth=2,
            color="r")
ridge_r2_array = np.array(ridge_r2_scores)
pl.errorbar(ranks, 
            ridge_r2_array.mean(axis=1),
            yerr=ridge_r2_array.std(axis=1) / np.sqrt(n_targets),
            elinewidth=2,
            linewidth=2,
            color="b")
pl.axis([0, max_rank, -1.1, 1.1])
pl.title("R2 scores")

# pl.figure(3)
# pl.clf()
# pl.plot(np.array(ridge_corr_scores), 'k-', lw=0.5)
# ridge_corr_array = np.array(ridge_corr_scores)
# pl.errorbar(ranks, 
#             ridge_corr_array.mean(axis=1),
#             yerr=ridge_corr_array.std(axis=1),
#             elinewidth=3,
#             linewidth=3,
#             color="r")
# pl.axis([0, max_rank, -.1, 1.1])
# pl.title("Ridge projected to low rank, corr scores")

# pl.figure(4)
# pl.clf()
# pl.plot(np.array(ridge_r2_scores), 'k-', lw=0.5)
# ridge_r2_array = np.array(ridge_r2_scores)
# pl.errorbar(ranks, 
#             ridge_r2_array.mean(axis=1),
#             yerr=ridge_r2_array.std(axis=1),
#             elinewidth=3,
#             linewidth=3,
#             color="r")
# pl.axis([0, max_rank, -.1, 1.1])
# pl.title("Ridge projected to low rank, r2 scores")

# pl.show()

