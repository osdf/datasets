import numpy as np
from scipy.ndimage import filters
from os.path import join
import h5py


from evaluate import evaluate
from dataset import _default_path, patch_x, patch_y
from vlfeat import vl_sift

_pshape = (patch_x, patch_y)

def store_as_sift(store, new_store, sigma, patch_shape=_pshape,
        midx=31.5, midy=31.5, scale=5.3):
    """Produce SIFT descriptors for patches in _store_ and
    save in _new_store_. Smooth patches with sigma, compute
    SIFT descriptor on reference frame [_midy_, _midx_, _scale_, 0].
    """
    frame = np.array([[midy, midx, scale, 0.]]).T

    for attrs in store.attrs:
        new_store.attrs[attrs] = store.attrs[attrs]

    new_store.attrs["frame"] = frame
    new_store.attrs["sigma"] = sigma
    
    for key in store.keys():
        if type(store[key]) is h5py.Group:
            grp = new_store.create_group(name=key)
            store_as_sift(store[key], grp, sigma, patch_shape, midx, midy, scale)
        if type(store[key]) is h5py.Dataset:
            dset = new_store.create_dataset(name=key, shape=(store[key].shape[0], 128), dtype=np.float64)
            siftify(store[key], dset, sigma, patch_shape, frame)


def siftify(store, dataset, sigma, patch_shape, frame):
    """SIFT descriptor for every patch in _store_.
    
    Smooth patches beforehand with gaussian of width _sigma_.
    Compute descriptor at frame _frame_.
    """
    totals = store.shape[0]
    print "Generating ", totals, " SIFT descriptors for", store, "with sigma", sigma
    for i in xrange(totals):
        patch = filters.gaussian_filter(store[i].reshape(patch_shape), sigma)
        [_, d] = vl_sift(patch.T, frames=frame)
        dataset[i] = np.asarray(d.ravel(), dtype=np.float64)


def find_sigma(dataset, log_sigmas, path=_default_path):
    """Find optimal sigma for SIFT descriptors.
    """
    fname = "".join(["evaluate_", dataset, "_64x64.h5"])
    store = h5py.File(join(path, fname))

    rocs = dict()
    dists = ["L2"]
    norms = ["id", "l2", "l1"]
    for sigma in log_sigmas:
        sift = "".join(["evaluate_", dataset, "_sift_", str(sigma), ".h5"])
        sift = h5py.File(join(path, sift), "w")
        store_as_sift(store, sift, sigma=np.exp(sigma))
        rocs[sigma] = evaluate(sift, distances=dists, normalizations=norms)
    return rocs
