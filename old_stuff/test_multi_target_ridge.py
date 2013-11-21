import numpy as np
import os
from ridge import _RidgeGridCV, _multi_corr_score, _multi_r2_score

ridge1 = _RidgeGridCV(alpha_min=1, alpha_max=1e6, 
                      n_grid_points=6, n_grid_refinements=2, logscale=True,
                      score_func=_multi_corr_score, cv=3)

import data
from delayed import make_delayed

X_train_raw = data.get_wordnet(mode="train")
X_train = make_delayed(X_train_raw, [2, 3, 4])
X_val_raw = data.get_wordnet(mode="val")
X_val = make_delayed(X_val_raw, [2, 3, 4])

Y_train = data.get_train()
Y_val = data.get_val()

ridge1.fit(X_train, Y_train)


from scipy import sparse

best_alphas = ridge1.best_alphas
betas_indep = ridge1.coef_.T

np.savez("/volatile/cache/indep_betas.npz",
         alphas=best_alphas, betas=betas_indep, scores=ridge1.best_mean_scores)

indep = np.load("/volatile/cache/indep_betas.npz")
best_alphas = indep["alphas"]
betas_indep = indep["betas"]

A = sparse.spdiags(best_alphas, [0], len(best_alphas), len(best_alphas))

mask = data.get_mask()

from sklearn.feature_extraction.image import grid_to_graph

n_x, n_y, n_z = mask.shape
connectivity = grid_to_graph(n_x=n_x, n_y=n_y, n_z=n_z, mask=mask)

from multi_task_ridge import _multi_target_ridge as multi_task_ridge

residual_norm = np.linalg.norm((X_train.dot(betas_indep) - Y_train).ravel(), 2)

print "Using the independent model, residual norm is %1.2f" % residual_norm

beta_old = betas_indep

cachedir = os.environ["DEFAULT_CACHE_DIR"]

for gamma in [0., 1., 5., 10., 50., 100., 500.]:

    print "evaluating gamma=%f" % gamma
    beta_new = multi_task_ridge(X_train, Y_train, 
                            M=connectivity, gamma=gamma,
                            A=A, alpha=1.,
                            warmstart=beta_old,
                            maxiter=51)

    y_pred_new = X_val.dot(beta_new)
    score_new = _multi_corr_score(Y_val, y_pred_new)

    filename = os.path.join(cachedir, "mt_ridge_gamma_%1.2f" % gamma)

    np.savez(filename, beta=beta_new, y_pred=y_pred_new, score=score_new)

    print "Some scores"
    print score_new[score_new.argsort()[::-1][:100]]

    beta_old = beta_new

