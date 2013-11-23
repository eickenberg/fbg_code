import numpy as np

import rank_k_bfgs


# Generate some data according to a noisy linear model

n_samples, n_features, n_targets, rank = 100, 80, 120, 10

SNR = 1e6

rng = np.random.RandomState(42)

# let's make it a Gaussian X first, but possibly we need to imitate
# the structure of our design matrix to perform a real test
X_train = rng.randn(n_samples, n_features)
U = rng.randn(n_features, rank)
V = rng.randn(n_targets, rank)

Y_train_clean = X_train.dot(U).dot(V.T)
signal_norms = np.sqrt((Y_clean ** 2).sum(axis=0))

Y_train_noisy = Y_train_clean + signal_norms / SNR * rng.randn(*Y_clean.shape)


ranks = np.arange(20) + 1
results = []
for r in ranks:
    result = rank_k_bfgs.rank_constrained_least_squares(X, Y_train_noisy, r,
                                                        1., 1.)
    results.append(result)

