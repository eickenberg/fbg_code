import numpy as np
import data
import scipy
import cortex

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
valdata_repeats = data.get_val(masked=ebamask, repeated=True)
valdata = data.get_val(masked=ebamask)

