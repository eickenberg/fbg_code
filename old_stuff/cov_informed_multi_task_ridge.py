import numpy as np

from scipy.sparse import coo_matrix
import data

Y_val_runs = data.get_val(repeats=True)
Y_val = data.get_val()

Y_val_runs_covariance_estimation_set = Y_val_runs[:90]
z_scored_Y_val_runs = ((Y_val_runs_covariance_estimation_set - 
                       Y_val_runs_covariance_estimation_set.mean(0)[np.newaxis, ...]) /
                       Y_val_runs_covariance_estimation_set.std(0)[np.newaxis, ...])

# Y_val_covariance_estimation_set = Y_val[:90]

# Will subtract precalculated/given mean from the different runs
# Y_val_noise = (Y_val_runs_covariance_estimation_set - 
#                Y_val_runs_covariance_estimation_set.mean(-1)[..., np.newaxis])
Y_val_noise = (z_scored_Y_val_runs - z_scored_Y_val_runs.mean(-1)[..., np.newaxis])


from covariance_on_connectivity import make_cov

connectivity = np.load(
    "/volatile/cache/sparse_connectivity_mesh.npy").item().tocsr()

set_diag = True
if set_diag:
    connectivity.setdiag(np.ones(connectivity.shape[0]))


cov_matrix_noise = make_cov(np.vstack(Y_val_noise.transpose(2, 0, 1)),
                            connectivity, correlation=False)

cov_matrix_responses = make_cov(np.vstack(z_scored_Y_val_runs.transpose(2, 0, 1)),
                                connectivity, correlation=False)


rcov = cov_matrix_responses
ncov = cov_matrix_noise

scov = (rcov - ncov).tocoo()


# # # compute Y_val_runs variance

# # rsd = np.sqrt((Y_val_runs.std(0) ** 2).mean(-1))

# # # divide by this variance to obtain correlations
# # # because scov has negative elements on the diagonal
# # row_vars = rsd[scov.row]
# # col_vars = rsd[scov.col]
# # corrs = scov.data / row_vars / col_vars

svar = scov.diagonal()
ssd = np.sqrt(svar)
row_vars = ssd[scov.row]
col_vars = ssd[scov.col]
corrs = scov.data / row_vars / col_vars

scorr = coo_matrix((corrs, (scov.row, scov.col)), shape=scov.shape).tocsr()

# # scorr.setdiag(np.zeros(scorr.shape[0]))


np.savez("/volatile/cache/sparse_covariances.npz", 
         ncov=cov_matrix_noise, 
         rcov=cov_matrix_responses,
         scov=scov,
         scorr=scorr)



cov_file = np.load("/volatile/cache/sparse_covariances.npz")

scov = cov_file['scov'].item().tocoo()
scorr = cov_file['scorr'].item().tocoo()




indep = np.load("/volatile/cache/indep_betas.npz")
best_alphas = indep["alphas"]
betas_indep = indep["betas"]

thresholded = True
alpha_rescaled = False
if thresholded:
    threshold_lower = -2.0
    threshold_upper = 0.80

    mask = np.logical_or(scorr.data > threshold_upper, scorr.data < threshold_lower)
    rows = scov.row[mask]
    cols = scov.col[mask]

    if alpha_rescaled:
        alphas_sqrt = np.sqrt(best_alphas)
        scale_row = alphas_sqrt[scorr.row[mask]]
        scale_col = alphas_sqrt[scorr.col[mask]]
        cdata = scorr.data[mask] * scale_row * scale_col
    else:
        cdata = scov.data[mask]
    M_matrix = coo_matrix((cdata, (rows, cols)), shape=scov.shape)
else:
    M_matrix = scov


from scipy import sparse

A = sparse.spdiags(best_alphas, [0], len(best_alphas), len(best_alphas))
from multi_task_ridge import _multi_target_ridge as multi_task_ridge
beta_old = betas_indep
# beta_old = np.load("/auto/k8/meickenberg/cache/thresh_-1.00_0.80_mt_ridge_with_corr_gamma_300.00.npz")['beta']

import os
cachedir = os.environ["DEFAULT_CACHE_DIR"]

from delayed import make_delayed
from ridge import _multi_corr_score

X_train_raw = data.get_wordnet(mode="train")
X_train = make_delayed(X_train_raw, [2, 3, 4])
X_val_raw = data.get_wordnet(mode="val")
X_val = make_delayed(X_val_raw, [2, 3, 4])

Y_train = data.get_train()
Y_val = data.get_val()

print "Starting loop"
import time

for gamma in [100, 500, 1000, 5000, 10000]:
    t = time.time()
    print "evaluating gamma=%f" % gamma
    beta_new = multi_task_ridge(X_train, Y_train, 
                            M=M_matrix, gamma=gamma,
                            A=A, alpha=1.,
                            warmstart=beta_old,
                            maxiter=61)

    y_pred_new = X_val.dot(beta_new)
    score_new = _multi_corr_score(Y_val, y_pred_new)

    filename = "mt_ridge_with_corr_gamma_%1.2f" % gamma
    if thresholded:
        filename = "thresh_%1.2f_%1.2f_" % (threshold_lower, threshold_upper) + filename
    if alpha_rescaled:
        filename = "rescaled_" + filename

    np.savez(os.path.join(cachedir, filename), beta=beta_new, y_pred=y_pred_new, score=score_new)

    print "Some scores"
    print score_new[score_new.argsort()[::-1][:100]]

    beta_old = beta_new
    print "run took %1.2f seconds" % (t - time.time())
