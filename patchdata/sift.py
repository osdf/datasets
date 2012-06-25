import numpy as np
from scipy.ndimage import filters
from os.path import join
import h5py


from evaluate import evaluate
from dataset import _default_path, patch_x, patch_y
from vlfeat import vl_sift


_pshape = (patch_x, patch_y)


def store_as_sift(store, new_store, patch_shape=_pshape,
        midx=31.5, midy=31.5, scale=0.):
    """Produce SIFT descriptors for patches in _store_ and
    save in _new_store_. Compute
    SIFT descriptor on reference frame [_midy_, _midx_, 2**_scale_, 0].
    """
    for attrs in store.attrs:
        new_store.attrs[attrs] = store.attrs[attrs]

    for key in store.keys():
        if type(store[key]) is h5py.Group:
            grp = new_store.create_group(name=key)
            store_as_sift(store[key], grp, patch_shape, midx, midy, scale)
        if type(store[key]) is h5py.Dataset:
            dset = new_store.create_dataset(name=key, shape=(store[key].shape[0], 128), dtype=np.float64)
            frame = np.array([[midy, midx, scale, 0.]]).T
            dset.attrs["frame"] = frame
            siftify(store[key], dset, patch_shape, frame)


def siftify(store, dataset, patch_shape, frame):
    """SIFT descriptor for every patch in _store_.
    
    Compute descriptor at frame _frame_.
    """
    totals = store.shape[0]
    print "Generating ", totals, " SIFT descriptors for", store, "at scale", frame[2]
    for i in xrange(totals):
        patch = store[i].reshape(patch_shape)
        [_, d] = vl_sift(patch.T, frames=frame)
        dataset[i] = np.asarray(d.ravel(), dtype=np.float64)


def find_scale(dataset, log_scales, path=_default_path):
    """Find optimal scale for SIFT descriptors.

    As of 2012/05/24, good scales are 1.75 and 2.1 (log-scale).
    """
    fname = "".join(["evaluate_", dataset, "_64x64.h5"])
    store = h5py.File(join(path, fname))

    rocs = dict()
    dists = ["L1", "L2", "COSINE"]
    norms = ["l1", "l2", "id"]
    for scale in log_scales:
        sift = "".join(["evaluate_", dataset, "_sift_", str(scale), ".h5"])
        sift = h5py.File(join(path, sift), "w")
        store_as_sift(store, sift, scale=2**scale)
        rocs[scale] = evaluate(sift, distances=dists, normalizations=norms)
    return rocs
