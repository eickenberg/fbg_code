import numpy as np
import tables
import os
from sklearn.externals.joblib import Memory

DATA_DIR = "/volatile/GallantLabVisit2013/"
cachedir = os.environ["DEFAULT_CACHE_DIR"]

templ = ["20101002AV", "20101009AV", "20101022AV"]

train_files = [os.path.join(DATA_DIR, "%s-R.hf5") % t for t in templ]
val_files = [os.path.join(DATA_DIR, "%s-P.hf5") % t for t in templ]

val_file_repeats = DATA_DIR + "AV-movie-val-repeats.hf5"

mask_file = DATA_DIR + "AV.hf5"

wn_stimuli = DATA_DIR + "WN-stimuli.hf5"
gabor_stimuli = DATA_DIR + "SN_movie_half_gabors.mat"

mem = Memory(cachedir=cachedir)


def get_mask():
    return tables.openFile(mask_file).getNode("/mask").read()\
        .astype(np.bool)


@mem.cache
def get_train(sessions=None, masked=True):
    """Retrieves training data for given sessions.
    Default: all sessions, mask applied"""

    if sessions is None:
        sessions = range(3)

    if isinstance(masked, bool) and masked:
        mask = get_mask()
        return np.concatenate(
            [tables.openFile(t).getNode('/data').read()[:, mask]
            for t in [train_files[i] for i in sessions]])
    elif isinstance(masked, np.ndarray):
	mask = masked
	return np.concatenate(
            [tables.openFile(t).getNode('/data').read()[:, mask]
            for t in [train_files[i] for i in sessions]])
    else:
        # raise NotImplementedError("This will exceed 4G of RAM")
        return np.concatenate(
            [tables.openFile(t).getNode('/data').read()
            for t in [train_files[i] for i in sessions]])


@mem.cache
def get_val(sessions=None, masked=True, repeats=False):
    """Retrieves training data for given sessions.
    Default: all sessions, mask applied"""

    if sessions is None:
        sessions = range(3)

    if repeats:
        if masked:
            return tables.openFile(val_file_repeats).getNode("/alldata").read()
        else:
            raise Exception("Repeats data is masked")
    else:
        if masked:
            mask = get_mask()
            return np.concatenate(
                [tables.openFile(t).getNode('/data').read()[:, mask]
                 for t in [val_files[i] for i in sessions]])
        else:
            raise NotImplementedError("This will exceed 4G of RAM")


def get_wordnet(mode="train", combined=True):

    if mode == "train":
        node = "Rstim"
    elif mode == "val":
        node = "Pstim"
    else:
        raise Exception("%s not understood" % mode)

    if combined:
        node = "comb%s" % node

    return tables.openFile(wn_stimuli).getNode("/" + node).read()


def get_gabor(mode="train", combined=True):

    if mode == "train":
        node = "Rstim"
    elif mode == "val":
        node = "Pstim"
    else:
        raise Exception("%s not understood" % mode)

    if combined:
        node = "comb%s" % node

    return tables.openFile(gabor_stimuli).getNode("/" + node).read()


nifti_files = [os.path.join(DATA_DIR, s) for s in
               ["2735201412413843298.nii.gz",
               "5775531250954080821.nii.gz",
               "17176994282695367364.nii.gz",
               "17231414115822952792.nii.gz",
               "12429498165058242653.nii.gz",
               "4427232006915868892.nii.gz",
               "8084543667400881630.nii.gz"]]

nifti_template = DATA_DIR + "AV_AV_huth_refepi.nii"
