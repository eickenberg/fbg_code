import numpy as np

import rank_k_bfgs
from ridge import _multi_corr_score, _multi_r2_score

# Generate some data according to a noisy linear model

n_samples, n_features, n_targets, rank = 200, 80, 120, 10

SNR = .1

rng = np.random.RandomState(42)

# let's make it a Gaussian X first, but possibly we need to imitate
# the structure of our design matrix to perform a real test
X_train = rng.randn(n_samples, n_features)
U = rng.randn(n_features, rank)
V = rng.randn(n_targets, rank)

Y_train_clean = X_train.dot(U).dot(V.T)
signal_norms = np.sqrt(
    ((Y_train_clean - Y_train_clean.mean(0)) ** 2).sum(axis=0))

noise_vector = rng.randn(*Y_train_clean.shape)
noise_vector /= np.sum((noise_vector ** 2).sum(0))

Y_train_noisy = Y_train_clean + signal_norms / SNR * noise_vector



num_test_samples = 50
test_SNR = 1e6
X_test = rng.randn(num_test_samples, n_features)
Y_test_clean = X_test.dot(U).dot(V.T)
test_signal_norms = np.sqrt((Y_test_clean ** 2).sum(axis=0))

Y_test_noisy = Y_test_clean + signal_norms / SNR * rng.randn(
    *Y_test_clean.shape)

max_rank = 20
ranks = np.arange(max_rank) + 1
results = []
corr_scores = []
r2_scores = []
for r in ranks:
    print "Rank: %d" % r
    U_, V_, something = result = \
        rank_k_bfgs.rank_constrained_least_squares(X_train, Y_train_noisy, r,
                                                        10., 10.)
    results.append(result)

    predictions = X_test.dot(U_).dot(V_.T)
    corr_scores.append(_multi_corr_score(Y_test_noisy, predictions))
    r2_scores.append(_multi_r2_score(Y_test_noisy, predictions))


import pylab as pl
pl.figure()
corr_array = np.array(corr_scores)
pl.plot(ranks, corr_array)
pl.errorbar(ranks, 
            corr_array.mean(axis=1), 
            yerr=corr_array.std(axis=1),
            elinewidth=3,
            linewidth=3,
            color="r")
pl.axis([0, max_rank, 0, 1.1])
pl.title("Correlation scores")

pl.figure()
pl.plot(np.array(r2_scores))
pl.axis([0, max_rank, 0, 1.1])
pl.title("R2 scores")


pl.show()
