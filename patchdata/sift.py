import numpy as np
from scipy.ndimage import filters
from os.path import join
import h5py


from dataset import _default_path
from vlfeat import vl_sift


def store_as_sift(store, new_store, sigma, patch_shape, midx=31.5, midy=31.5, scale=5.3):
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
    print "Generating ", totals, " SIFT descriptors for", store
    for i in xrange(totals):
        patch = filters.gaussian_filter(store[i].reshape(patch_shape), sigma)
        [_, d] = vl_sift(patch.T, frames=frame)
        dataset[i] = np.asarray(d.ravel(), dtype=np.float64)
