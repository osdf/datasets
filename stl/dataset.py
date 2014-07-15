"""
This module provides handling
of the basic aspects of the patchdata set:
    - building hdf5 stores (for the original sized patches)
      in particular the stores to evaluate descriptors
    - resizing patches (works only on available stores)
    - randomly selecting patches
    - printing general information

Start with build_store(origin="train", consecutive=False)
same for origin="test". Then merge_train_test(). And
then gray_store() on the merged store. (No merging if
you need to keep train_rgb and test_rgb seperated --
then simply run gray_store on both). Don't forget to
resize and normalize properly.
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


def pair_store(store, fname, pairs):
    """
    Generate positive (same class) and negative pairs of images.
    """
    from itertools import combinations as comb

    totals = 10*pairs

    print "Pairing on store", store
    print "Positive Pairs/Negative Pairs", totals, totals

    store = get_store(fname=store)
    inpts = store['inpts']
    lbls = store['trgts']

    positives = []
    classes = {}
    # stl-10 has 10 classes
    for c in xrange(10):
        idx = np.argwhere(lbls[:]==c).ravel()
        classes[c] = idx
        cnt = 0
        while True:
            p1 = np.random.randint(idx.shape[0])
            p2 = np.random.randint(idx.shape[0])
            if p1 == p2:
                continue
            # get actual indices
            p1 = idx[p1]
            p2 = idx[p2]

            if p1 > p2:
                p1, p2 = p2, p1
            
            if (p1, p2) in positives:
                continue
            
            positives.append((p1, p2))
            
            cnt = cnt + 1
            if cnt == pairs:
                print "Done with class", c
                print "Current shape of positives:", len(positives)
                break
    # final shuffle
    helpers.shuffle_list(positives)

    negatives = []
    for t in xrange(totals):
        # get c1, c2, two different classes
        c1 = np.random.randint(10)
        c2 = np.random.randint(10)
        while True:
            if c1 != c2:
                break
            c2 = np.random.randint(10)
        while True:
            n1 = np.random.randint(classes[c1].shape[0])
            n2 = np.random.randint(classes[c2].shape[0])
            n1 = classes[c1][n1]
            n2 = classes[c2][n2]

            if n1 > n2:
                n1, n2 = n2, n1
            
            if (n1, n2) not in negatives:
                negatives.append((n1, n2))
                break
    print "Negatives are", len(negatives)
    # Check if everything is correct:
    # Positives: Pairs of same class, ordered, no duplicates.
    print "Check pairings ..."
    for p in positives:
        p1, p2 = p
        assert lbls[p1] == lbls[p2], "Different classes %d, %d"%(lbls[p1], lbls[p2]) 
        if p1 > p2:
            print "MISTAKE: p1 > p2", p1, p2
            assert False
    _tmp = set(positives)
    assert len(_tmp) == len(positives), "Duplicates in positive?"
    print "Positives ok ..."
    for n in negatives:
        n1, n2 = n
        assert lbls[n1] != lbls[n2], "Same classes %d, %d"%(lbls[n1], lbls[n2]) 
        if n1 > n2:
            print "MISTAKE: n1 > n2", n1, n2
            assert False
    _tmp = set(negatives)
    assert len(_tmp) == len(negatives), "Duplicates in negatives?"
    print "Negatives ok ..."

    print "Write to h5."
    # open up hdf5 file
    h5f = h5py.File(fname, "w")
    print "Created hdf5 file", h5f
    grp = h5f.create_group("train")
    grp.attrs["patch_shape"] = (patch_x, patch_y)

    ins = grp.create_dataset(name="inpts", shape=(4*totals, patch_x*patch_y), dtype=np.float32)
    ins.attrs["patch_shape"] = (patch_x, patch_y)
    trgts = grp.create_dataset(name="trgts", shape=(4*totals,), dtype=np.int)
    
    for t in xrange(totals):
        p1, p2 = positives[t]
        ins[4*t,:] = inpts[p1]
        ins[4*t+1,:] = inpts[p2]
        trgts[4*t] = 1
        trgts[4*t+1] = 1
        n1, n2 = negatives[t]
        ins[4*t+2,:] = inpts[n1]
        ins[4*t+3,:] = inpts[n2]
        trgts[4*t+2] = 0
        trgts[4*t+3] = 0
    h5f.close()


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
