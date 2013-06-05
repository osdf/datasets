"""
This module provides some handling for the van hateren dataset:
    - building hdf5 stores (for extracted patches!)
"""


import subprocess
import random
from itertools import izip, product
from collections import defaultdict
from os.path import dirname, join, exists
import hashlib
import numpy as np
import array
import h5py
import glob


try:
    import Image as img
except:
    import PIL as img


from helpers import helpers

# size of an original image in pixels
patch_x = 1024
patch_y = 1536

# dont crop pictures from the border
border_x = 20
border_y = 20

_default_path = dirname(__file__)


def get_store(fname="vanhateren.h5", path=_default_path):
    print "Loading from store", fname
    return h5py.File(join(path, fname), 'r')


def build_store(fname="vanhateren.h5", path=_default_path, patch_freq=400,
        patch_shape=(16,16)):
    """
    """
    print "Writing to", fname
    f = h5py.File(join(path, fname), "w")
    all_images = get_images("iml")
    
    grp = f.create_group("train")
    size = len(all_images) * patch_freq
    dset = grp.create_dataset(name="inputs", shape=(size, patch_shape[0]*patch_shape[1]), dtype=np.float64)
    count = 0

    for j, image in enumerate(all_images):
        handle = open(image, 'rb')
        s = handle.read()
        arr = array.array('H', s)
        arr.byteswap()
        im = np.array(arr, dtype=np.float).reshape(1024, 1536)
        mx = im.max() * 1.
        im /= mx
        # get random x/y positions in img to crop patches
        xrand = np.floor(np.random.rand(patch_freq) * (1024 - 2*border_x - 2*patch_shape[0]) + border_x + patch_shape[0])
        yrand = np.floor(np.random.rand(patch_freq) * (1536 - 2*border_y - 2*patch_shape[1]) + border_y + patch_shape[1])

        # crop patches
        for x, y in izip(xrand, yrand):
            dset[count, :] = im[x-patch_shape[0]/2:x+patch_shape[0]/2, y-patch_shape[1]/2:y+patch_shape[1]/2].flatten()
            count += 1
    # to be in accordance with patchdata set
    grp = f.create_group("validation")
    dset1 = grp.create_dataset(name="inputs", shape=(100, patch_shape[0]*patch_shape[1]), dtype=np.float64)
    dset1[:,:] = dset[:100, :]
    f.attrs["patch_shape"] = (patch_y, patch_x)
    print "Collected patches:", count
    f.close()


def get_images(path):
    """
    """
    return glob.glob(join(path, "*.iml"))


def crop_store(store, x, y, dx, dy, cache=False):
    """A new store that contains cropped images from _store_.
    _x_, _y_, _dx_ and _dy_ are the cropping parameters.
    """
    print "Cropping from store", store, "with (x,y);(dx, dy)", x, y, dx, dy
    sfn = store.filename.split(".")[0]
    name = hashlib.sha1(sfn + str(x) + str(y) + str(dx) + str(dy))

    name = name.hexdigest()[:8] + ".crop.h5"
    if cache is True and exists(name):
        print "Using cached version ", name
        return h5py.File(name, 'r+')

    print "No cache, writing to", name
    crop = h5py.File(name, 'w')
    helpers.crop(store, crop, x, y, dx, dy)
    crop.attrs["Cropped"] = "from " + str(store.filename)
    return crop


def stationary_store(store, eps=1e-8, C=1., div=1., chunk=512, cache=False):
    """A new store that contains stationary images from _store_.
    """
    print "Stationarize store", store, "with eps, C, div" , eps, C, div
    sfn = store.filename.split(".")[0]
    name = hashlib.sha1(sfn + str(C) + str(eps) + str(chunk))
    name = name.hexdigest()[:8] + ".stat.h5"
    if cache is True and exists(name):
        print "Using cached version ", name
        return h5py.File(name, 'r+')

    print "No cache, writing to", name
    stat = h5py.File(name, 'w')
    helpers.stationary(store, stat, chunk=chunk, eps=eps, C=C, div=div)
    stat.attrs["Stationary"] = "from " + str(store.filename)
    return stat


def row0_store(store, chunk=512, cache=False):
    """A new store that contains 0-mean images from _store_.
    """
    print "Row0 store", store
    sfn = store.filename.split(".")[0]
    name = hashlib.sha1(sfn + str(chunk))
    name = name.hexdigest()[:8] + ".row0.h5"
    if cache is True and exists(name):
        print "Using cached version ", name
        return h5py.File(name, 'r+')

    print "No cache, writing to", name
    r0 = h5py.File(name, 'w')
    helpers.row0(store, r0, chunk=chunk)
    r0.attrs["Row0"] = "from " + str(store.filename)
    return r0 


def feat0_store(store, to_sub, chunk=512, cache=False):
    """A new store that is featurewise 0-mean.
    """
    print "Feat0 store", store
    sfn = store.filename.split(".")[0]
    name = hashlib.sha1(sfn + str(to_sub) + str("feat0_store") + str(chunk))
    name = name.hexdigest()[:8] + ".feat0.h5"
    if cache is True and exists(name):
        print "Using cached version ", name
        return h5py.File(name, 'r+')

    print "No cache, writing to", name
    f0 = h5py.File(name, 'w')
    helpers.feat_sub(store, f0, chunk=chunk, sub=to_sub)
    f0.attrs["Feat0"] = "from " + str(store.filename)
    return f0 


def feat_std1_store(store, to_div, chunk=512, cache=False):
    """
    New store has standard deviation 1 per feature.
    """
    print "Feat_std1 store", store
    sfn = store.filename.split(".")[0]
    name = hashlib.sha1(sfn + str(to_div) + str("feat_std1_store") + str(chunk))
    name = name.hexdigest()[:8] + ".feat_std1.h5"
    if cache is True and exists(name):
        print "Using cached version ", name
        return h5py.File(name, 'r+')

    print "No cache, writing to", name
    fstd1 = h5py.File(name, 'w')
    helpers.feat_div(store, fstd1, chunk=chunk, div=to_div)
    fstd1.attrs["Feat_std1"] = "from " + str(store.filename)
    return fstd1


def gstd1_store(store, to_div, chunk=512, cache=False):
    """A new store that has global std = 1.
    """
    print "GStd1 store", store
    sfn = store.filename.split(".")[0]
    name = hashlib.sha1(sfn + str(to_div) + str("gstd1_store") + str(chunk))
    name = name.hexdigest()[:8] + ".gstd1.h5"
    if cache is True and exists(name):
        print "Using cached version ", name
        return h5py.File(name, 'r+')

    print "No cache, writing to", name
    std = h5py.File(name, 'w')
    helpers.global_div(store, std, chunk=chunk, div=to_div)
    std.attrs["GStd1"] = "from " + str(store.filename)
    return std


def resize_store(store, shape, cache=False):
    """A new store that contains stationary images from _store_.
    """
    print "Resizing store", store, "to new shape", shape
    sfn = store.filename.split(".")[0]
    name = hashlib.sha1(sfn + str(shape))
    name = name.hexdigest()[:8] + ".resz.h5"
    if cache is True and exists(name):
        print "Using cached version ", name
        return h5py.File(name, 'r+')

    print "No cache, writing to", name
    rsz = h5py.File(name, 'w')
    helpers.resize(store, rsz, shape)
    rsz.attrs["Resized"] = "from " + str(store.filename)
    return rsz


def at_store(store, M, chunk=512, cache=False):
    """
    """
    print "AT store", store
    sfn = store.filename.split(".")[0]
    name = hashlib.sha1(sfn + str(M) + str(chunk))
    name = name.hexdigest()[:8] + ".at.h5"
    if cache is True and exists(name):
        print "Using cached version ", name
        return h5py.File(name, 'r+')

    print "No cache, writing to", name
    at = h5py.File(name, 'w')
    helpers.at(store, at, M, chunk=chunk)
    at.attrs["AT"] = "from " + str(store.filename)
    return at


def zeroone_store(store, chunk=512, cache=False):
    """
    """
    print "Zeroone store", store
    sfn = store.filename.split(".")[0]
    name = hashlib.sha1(sfn+str(chunk))
    name = name.hexdigest()[:8] + ".zo.h5"

    if cache is True and exists(name):
        print "Using cached version ", name
        return h5py.File(name, 'r+')

    print "No cache, writing to", name
    zo = h5py.File(name, 'w')
    helpers.zeroone(store, zo, chunk=chunk)
    zo.attrs["ZO"] = "from " + str(store.filename)
    return zo 


def zeroone_group(store, chunk=512, group=["match", "non-match"], cache=False):
    """
    """
    print "Zeroone Group", store, group
    sfn = store.filename.split(".")[0]
    name = hashlib.sha1(sfn+str(chunk)+"zeroone_group"+str(group))
    name = name.hexdigest()[:8] + ".zogrp.h5"

    if cache is True and exists(name):
        print "Using cached version ", name
        return h5py.File(name, 'r+')

    print "No cache, writing to", name
    zo = h5py.File(name, 'w')
    helpers.zeroone_group(store, zo, group=group, chunk=chunk)
    zo.attrs["ZO_GRP"] = "from " + str(store.filename)
    return zo 


def _crop_to_numpy(patchfile, ravel=True):
    """Convert _patchfile_ to a numpy array with patches per row.

    A _patchfile_ is a .bmp file.
    """
    patches = img.open(patchfile)
    arr = [] 
    for row, col in product(xrange(rows), xrange(cols)):
        ptch = patches.crop((col*patch_x, row*patch_y, (col+1)*patch_x, (row+1)*patch_y))
        if ravel:
            arr.append(np.array(ptch).ravel())
        else:
            arr.append(np.array(ptch))
    return np.array(arr)


def _fill_up(from_store, to_store, indices, pos, chunk):
    """Fill _to_store_ with patches from _from_store_.
    """
    for i in xrange(0, len(indices), chunk):
        _idx = indices[i:i+chunk]
        to_store[pos:pos+len(_idx)] = from_store[_idx]
        pos += len(_idx)
    return pos


def _available_patches(dataset):
    """Number of patches in _dataset_ (a path).
    
    Only available through the line count 
    in info.txt -- use unix 'wc'.
    
    _path_ is supposed to be a path to
    a directory with bmp patchsets.
    """
    fname = join(dataset, "info.txt")
    p = subprocess.Popen(['wc', '-l', fname],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    result, err = p.communicate()
    if p.returncode != 0:
        raise IOError(err)
    return int(result.strip().split()[0])


def _build_pairing_store(group, name, pairings, store):
    """Create a dataset with pairs from _pairings_.
    
    The dataset _name_ is below group _group_.  _pairings are 
    pairs of indices that refer to _store_. The new store 
    (a hdf5 dataset) is a flat array, yet, blocks of 
    two consecutive rows form a pair.

    Note: The new store has dtype=np.float64.
    """
    dset = group.create_dataset(name=name, shape=(2*len(pairings), patch_x*patch_y), dtype=np.float64)
    dset.attrs["patch_shape"] = (patch_y, patch_x)
 
    for j, pair in enumerate(pairings):
        p1, p2 = _patches_from_pair(pair=pair, store=store)
        dset[2*j] = p1
        dset[2*j+1] = p2


def _patches_from_pair(pair, store):
    return store[pair[0],:], store[pair[1],:]


if __name__=="__main__":
    build_store()
