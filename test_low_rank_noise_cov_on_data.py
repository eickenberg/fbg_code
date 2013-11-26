import numpy as np
import data
import scipy
import cortex

from multi_target_ridge_with_noise_covariance import \
    get_inv_diag_plus_low_rank_cov_op, MultiTaskRidge

numtime = 1000


# load stimuli
trnstim = data.get_wordnet("train")
valstim = data.get_wordnet("val")

delays = [2, 3, 4]
deltrnstim = np.hstack([np.roll(trnstim, d, 0) for d in delays])
delvalstim = np.hstack([np.roll(valstim, d, 0) for d in delays])

sdeltrnstim = scipy.sparse.csr_matrix(deltrnstim)

ebamask = cortex.get_roi_mask("MLfs", "20121210ML_auto1", roi="EBA")["EBA"] > 0

trndata = data.get_train(masked=ebamask)
# use first block for noise covariance estimation
valdata_repeats = data.get_val(masked=ebamask, repeats=True)[:90]
# use second and third block for evaluation
valdata = data.get_val(masked=ebamask)[90:]

# zscore it?
valdata_repeats = ((valdata_repeats -
                   valdata_repeats.mean(0)[np.newaxis, ...]) /
                   valdata_repeats.std(0)[np.newaxis, ...])

valdata_noise = valdata_repeats - valdata_repeats.mean(-1)[..., np.newaxis]

inv_noise_cov = get_inv_diag_plus_low_rank_cov_op(
    np.vstack(valdata_noise.transpose(2, 0, 1)))

