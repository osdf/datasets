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
# color images
channels = 3

# number of classes
classes = 10

# number of training images per class
train_size = 500

# number of test images
test_size = 800

# standard path
_default_path = dirname(__file__)
_bin_path = dirname(__file__) + "./stl10_binary"


def get_store(fname="stl_96x96_train.h5", path=_default_path, 
        verbose=True, access='r'):

    if verbose:
        print "Loading from store", fname
    return h5py.File(join(path, fname), access)


def build_store(origin="train", path=_bin_path, consecutive=True):
    """
    """
    if origin is "train":
        if consecutive:
            fname = "stl_96x96_train.h5"
        else:
            fname = "stl_96x96_train_rgb.h5"
        size = train_size
        inpts = "train_X.bin"
        lbls = "train_y.bin"
    elif origin is "test":
        if consecutive:
            fname = "stl_96x96_test.h5"
        else:
            fname = "stl_96x96_test_rgb.h5"
        size = test_size
        inpts = "test_X.bin"
        lbls = "test_y.bin"
    totals = classes * size
    
    print "Writing to", fname
    h5f = h5py.File(fname, "w")

    # data is available in binary fromat
    f = open(join(path, inpts), "rb")
    
    dset = h5f.create_dataset(name="inpts", shape=(totals,\
            channels*patch_x*patch_y), dtype=np.uint8)
    tmp = np.zeros((channels, patch_x * patch_y), dtype=np.uint8)

    for i in xrange(totals):
        for c in xrange(channels):
            uints = f.read(patch_x*patch_y)
            if (len(uints) != patch_x*patch_y):
                print "ERROR: in 'build_store', expected more data."
                print "ERROR: remove the generated file: ", fname
                h5f.close()
                return
            tmp[c, :] = np.frombuffer(uints, dtype=np.uint8)
            tmp[c, :] = tmp[c].reshape(patch_x, patch_y).T.ravel()
            if consecutive:
                dset[i, :] = tmp.ravel()
            else:
                dset[i, :] = tmp.T.ravel()
    f.close()
    
    f = open(join(path, lbls), "rb")
    cset = h5f.create_dataset(name="trgts",\
            shape=(totals,), dtype=np.int)
    for i in xrange(totals):
        lbl = f.read(1)
        if (len(lbl) != 1):
            print "ERROR: in 'build_store', expected more data."
            print "ERROR: remove the generated file: ", fname
            h5f.close()
            return
        cset[i] = ord(lbl) - 1
    f.close()
 
    h5f.attrs["stl"] = origin 
    h5f.attrs["patch_shape"] = (patch_y, patch_x)
    h5f.attrs["channels"] = channels
    h5f.close()
    print "Wrote store to", fname


def merge_train_test():
    """
    Merges train and test store.
    """
    trn = "stl_96x96_train_rgb.h5"
    tst = "stl_96x96_test_rgb.h5"
    fname = "stl_96x96_merged_rgb.h5"

    print "Writing to", fname
    trn = h5py.File(trn, "r")
    tst = h5py.File(tst, "r")

    merged = h5py.File(fname, "w")
    totals_trn = trn['inpts'].shape[0]
    totals_tst = tst['inpts'].shape[0]
    totals = totals_trn + totals_tst

    print "Merged data has size", totals

    dset = merged.create_dataset(name="inpts", shape=(totals,\
            channels*patch_x*patch_y), dtype=np.uint8)
    cset = merged.create_dataset(name="trgts",\
            shape=(totals,), dtype=np.int)

    cur = 0
    for i in xrange(totals_trn):
        dset[cur, :] = trn['inpts'][i]
        cset[cur] = trn['trgts'][i]
        cur = cur + 1

    for i in xrange(totals_tst):
        dset[cur, :] = tst['inpts'][i]
        cset[cur] = tst['trgts'][i]
        cur = cur + 1

    merged.attrs["stl"] = "merged"
    merged.attrs["patch_shape"] = (patch_y, patch_x)
    merged.attrs["channels"] = channels
    merged.close()
    print "Wrote store to", fname


def gray_store(store="stl_96x96_train.h5",
        patch_x=patch_x, patch_y=patch_y):
    """
    """
    print "Note that you need a rgb ordering of the pixels!"
    print "Therefore we add a suffix '_rgb'."
    store = store.split('.')
    fname = store[0] + "_gray." + store[1]
    store = store[0] + "_rgb." + store[1]
    
    print "And now reading from {0}".format(store)
    store = get_store(fname=store) 

    print "Writing to", fname
    h5f = h5py.File(fname, "w")

    totals = store['inpts'].shape[0]
    dset = h5f.create_dataset(name="inpts", shape=(totals,\
            patch_x*patch_y), dtype=np.float)
    tmp = np.zeros((channels, patch_x * patch_y), dtype=np.float)
    for i in xrange(totals):
        tmp = store['inpts'][i].reshape(patch_x, patch_y, channels)
        dset[i, :] = np.dot(tmp[:, :, :3], [0.299, 0.587, 0.144]).ravel()
    dset.attrs["patch_shape"] = (96, 96)

    cset = h5f.create_dataset(name="trgts",\
            shape=(totals,), dtype=np.int)
    for i in xrange(totals):
        cset[i] = store["trgts"][i] 

    h5f.attrs["GrayStore"] = "from " + str(store.filename)

    h5f.close()
    print "Wrote store to", fname


def pair_store(store, ):
    """
    Generate positive (same class) and negative pairs of images.

    get all indices of class i.
    shuffle every list.
    for every class i, get all x-many 2 pairs of indices.
    randomly zip over all class i pairs.
    """
    from itertools import combinations as comb

    print "Pairing on store", store
    store = get_store(fname=store)
    inpts = store['intps']
    lbls = store['trgts']

    # stl-10 has 10 classes
    positives = []
    classes = {}
    for c in xrange(10):
        idx = np.argwhere(lbls[:]==c).ravel()
        classes[c] = idx
        # shuffle indices?
        pairs_c = comb(idx, 2)
        for j, p in enumerate(pairs_c):
            positives.append(p)
    # shuffle positives
    negatives = []
    for t in xrange(totals):
        # get c1, c2, two different classes
        c1 = np.random.random_integers(10)
        c2 = 0
        while True:
            c2 = np.random.random_integers(10)
            if c1 != c2:
                break
        while True:
            n1 = sample(classes[c1].shape[0])
            n2 = sample(classes[c2].shape[0])
            if (n1, n2) not in negatives:
                negatives.append((n1, n2))
                break
    # open up hdf5 file
    # make group train
    # make two datasets, inpts, trgts
    for t in xrange(totals):
        p1, p2 = positives[t]
        ins[4*t,:] = inpts[p1]
        ins[4*t+1,:] = store[p2]
        trgts[4*t] = 1
        trgts[4*t+1] = 1
        n1, n2 = negatives[t]
        ins[4*t+2,:] = inpts[n1]
        ins[4*t+3,:] = inpts[n2]
        trgts[4*t+2] = 0
        trgts[4*t+3] = 0


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
