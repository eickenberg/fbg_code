import numpy as np
import data
import cortex
import scipy.sparse
from sklearn.cluster import WardAgglomeration
from sklearn.feature_extraction import image
from rank_k_bfgs import rank_constrained_least_squares, get_vec_and_grad_func
import matplotlib.pyplot as plt

numtime = 1000
numclusters = 100

# Load stimuli
trnstim = data.get_gabor("train")
valstim = data.get_gabor("val")

delays = [2,3,4]
deltrnstim = np.hstack([np.roll(trnstim, d, 0) for d in delays])
delvalstim = np.hstack([np.roll(valstim, d, 0) for d in delays])

sdeltrnstim = scipy.sparse.csr_matrix(deltrnstim)

# Select some voxels
cort_mask = cortex.get_cortical_mask("MLfs", "20121210ML_auto1", "thick")
rois = ["V1", "V2", "V3"]
masks = [cortex.get_roi_mask("MLfs", "20121210ML_auto1", roi=roi)[roi] > 0 for roi in rois]
roimask = reduce(lambda x, y: (x + y), masks)
wardmask = cort_mask-roimask

# Load training, test fMRI data
trndata_roi = data.get_train(masked=roimask)
trndata_ward = data.get_train(masked=wardmask)

connectivity = image.grid_to_graph(n_x=wardmask.shape[0], n_y=wardmask.shape[1],
                                   n_z=wardmask.shape[2], mask=wardmask)
ward = WardAgglomeration(n_clusters=numclusters, connectivity=connectivity,
                         memory='nilearn_cache')
ward.fit(trndata_ward)
labels = ward.labels_
trndata_collapsed = np.array([data[:, labels == i].mean(1) for i in range(numclusters)])
trndata = np.hstack((trndata_roi, trndata_collapsed))
valdata = data.get_val(masked=roimask)

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
