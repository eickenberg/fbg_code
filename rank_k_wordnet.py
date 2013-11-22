import numpy as np
import data
import cortex
import scipy.sparse
from rank_k_bfgs import rank_constrained_least_squares, get_vec_and_grad_func
import matplotlib.pyplot as plt

numtime = 1000

# Load stimuli
trnstim = data.get_wordnet("train")
valstim = data.get_wordnet("val")

delays = [2,3,4]
deltrnstim = np.hstack([np.roll(trnstim, d, 0) for d in delays])
delvalstim = np.hstack([np.roll(valstim, d, 0) for d in delays])

sdeltrnstim = scipy.sparse.csr_matrix(deltrnstim)

# Select some voxels
ebamask = cortex.get_roi_mask("MLfs", "20121210ML_auto1", roi="EBA")["EBA"] > 0

# Load training, test fMRI data
trndata = data.get_train(masked=ebamask)
valdata = data.get_val(masked=ebamask)

# Run some shit
maxit = 500
results = []
for r in [1, 5, 10, 20, 100]: #range(1, 6):
    print "\r\nCase: r = %i" % r
    energies = []
    
    def cb(energy):
        # callback_env["it"] += 1
        #energy = func(vec)
        #print "\titer %03i/%03i: energy = %g" % (
        #    callback_env["it"], maxit, energy)
        #print energy
        energies.append(energy)
        # print func(vec)

    U, V, res = rank_constrained_least_squares(
        sdeltrnstim[:numtime], trndata[:numtime], r,
        alpha1=100. * numtime / r / sdeltrnstim.shape[1],
        alpha2=100. * numtime / r / trndata.shape[1],
        m=10,  # memory budget
        max_bfgs_iter=maxit,)
        #callback=cb)

    results.append((U, V, res))
    #plt.plot(, label="r=%i" % r)
    #plt.yscale("log")
