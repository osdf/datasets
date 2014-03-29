"""
This module provides handling
of the basic aspects of the patchdata set:
    - building hdf5 stores (for the original sized patches)
      in particular the stores to evaluate descriptors
    - resizing patches (works only on available stores)
    - randomly selecting patches
    - printing general information
"""


import subprocess
import random
from itertools import izip, product
from collections import defaultdict
from os.path import dirname, join, exists
import hashlib
import numpy as np
import h5py

try:
    import Image as img
except:
    import PIL as img


from helpers import helpers


# size of an original patch in pixels
patch_x = 96
patch_y = 96

# helpful defaults
_default_path = dirname(__file__)
_default_pairings = (500, 1000, 2500, 5000, 10000, 25000)


def get_store(fname="stl_96x96.h5", path=_default_path, verbose=True, access='r'):
    if verbose:
        print "Loading from store", fname
    return h5py.File(join(path, fname), access)


def build_store(fname="stl_96x96.h5", path=_default_path, dataset=dataset):
    print "Writing to", fname
    f = h5py.File(join(path, fname), "w")
    for ds in dataset:
        ds_path = join(path, ds)
        totals = _available_patches(ds_path)
        dset = f.create_dataset(name=ds, shape=(totals, patch_x*patch_y), dtype = np.uint8)
        bmps, mod = divmod(totals, per_bmp)
        print "For", ds, "reading a total of", totals, "patches from", bmps, "files into", fname
        for i in xrange(bmps):
            bmp = join(ds_path, ''.join(["patches", str(i).zfill(4), ".bmp"]))
            dset[i*per_bmp:(i+1)*per_bmp] = _crop_to_numpy(bmp)
        if mod > 0:
            bmp = join(ds_path, ''.join(["patches", str(bmps).zfill(4), ".bmp"])) 
            dset[-(totals%per_bmp):] = _crop_to_numpy(bmp)[:mod]
    f.attrs["dataset"] = dataset
    f.attrs["patch_shape"] = (patch_y, patch_x)
    f.close()
    print "Wrote", dataset, "to", fname, f


def build_pairs_store(store, pair_list=_default_pairings, path=_default_path, tag=None):
    """Put matching/non-matching pairs into a hdf5.

    There is one hdf5 per dataset, with the available number of pairs 
    in _pair_list_. These numbers form the groups of every store. 
    The store is using the original patches, scaled versions of 
    these patches should be generated with resize.
    
    Every group has two datasets, 'match' and 'non-match',
    both arrays of equal length, the original pairs are formed by
    blocks of consecutive rows.

    Note: Every dataset in this store has dtype=np.float64, to ease
    later evaluation (avoid costly rebuilding everyting into float64).
    For this reason, reconsider when 'extending' pair_list into the 100.000
    range.
    """
    if tag is None:
        tag = ""
    else:
        tag = "".join([tag, "_"])
    for ds in dataset:
        fname = "".join([tag, "evaluate_", ds, "_", "64x64.h5"])
        print "\nBuilding evaluate store", fname, "for", pair_list
        f = h5py.File(join(path, fname), "w")
        for pairs in pair_list:
            grp = f.create_group(name=str(pairs))
            mtch, non_mtch, ids = matches(ds, pairs, path=path)
            _build_pairing_store(group=grp, name="match", pairings=mtch, store=store[ds])
            _build_pairing_store(group=grp, name="non-match", pairings=non_mtch, store=store[ds])
        f.attrs["dataset"] = ds
        f.attrs["patch_shape"] = (patch_y, patch_x)
        f.attrs["pairs"] = pair_list
        f.close()


def stationary_store(store, eps=1e-8, C=1., div=1., chunk=512, cache=False, exclude=[None], verbose=True):
    """A new store that contains stationary images from _store_.
    """
    if verbose:
        print "Stationarize store", store, "with eps, C, div" , eps, C, div
    sfn = store.filename.split(".")[0]
    name = hashlib.sha1(sfn + str(C) + str(eps) + str(chunk))
    name = name.hexdigest()[:8] + ".stat.h5"
    if cache is True and exists(name):
        print "Using cached version ", name
        return h5py.File(name, 'r+')

    print "No cache, writing to", name
    stat = h5py.File(name, 'w')
    helpers.stationary(store, stat, chunk=chunk, eps=eps, C=C, div=div, exclude=exclude)
    stat.attrs["Stationary"] = "from " + str(store.filename)
    return stat


def resize_store(store, shape, cache=False, exclude=[None], verbose=True):
    """A new store that contains resized images from _store_.
    """
    if verbose:
        print "Resizing store", store, "to new shape", shape
    sfn = store.filename.split(".")[0]
    name = hashlib.sha1(sfn + str(shape))
    name = name.hexdigest()[:8] + ".resz.h5"
    if cache is True and exists(name):
        if verbose: print "Using cached version ", name
        return h5py.File(name, 'r+')

    print "No cache, writing to", name
    rsz = h5py.File(name, 'w')
    helpers.resize(store, rsz, shape, exclude=exclude)
    rsz.attrs["Resized"] = "from " + str(store.filename)
    return rsz


if __name__=="__main__":
    build_store()
