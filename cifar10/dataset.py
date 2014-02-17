"""
Handle data from CIFAR 10.

"""


import h5py
import cPickle
from os.path import dirname, join, exists
import hashlib

import numpy as np
try:
    import Image as img
except:
    import PIL as img


from helpers import helpers


_default_path = dirname(__file__)
_default_name = join(_default_path, "cifar10_32x32.h5")
_default_gray = join(_default_path, "cifar10_gray_32x32.h5")
_batch_path = "cifar-10-batches-py"
_train = ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4"]
_valid = ["data_batch_5"]
_test  = ["test_batch"]
_batch_size = 10000
_all = _train + _valid + _test


def get_store(fname=_default_name, path=_default_path, verbose=True):
    if verbose:
        print "Loading from store", fname
    return h5py.File(join(path, fname), 'r')


def build_store(store=_default_name, size=(32, 32)):
    """Build a hdf5 data store for CIFAR.
    """
    print "Writing to", store
    h5file = h5py.File(store, "w")

    _create_grp(store=h5file, grp_name="train", batches=_train, size=size)
    _create_grp(store=h5file, grp_name="validation", batches=_valid, size=size)
    _create_grp(store=h5file, grp_name="test", batches=_test, size=size)

    print "Closing", store
    h5file.close()


def build_gray_store(store=_default_gray, size=(32, 32)):
    """Build a hdf5 data store for CIFAR10, inputs are gray.
    """
    print "Writing to", store
    h5file = h5py.File(store, "w")

    _create_grp(store=h5file, grp_name="train", batches=_train, gray=True, size=size)
    _create_grp(store=h5file, grp_name="validation", batches=_valid, gray=True, size=size)
    _create_grp(store=h5file, grp_name="test", batches=_test, gray=True, size=size)

    print "Closing", store
    h5file.close()


def build_gray_seq(store=_default_gray, base_sz=(11, 11), seq_len=3, delta=4):
    """
    """
    sizex = base_sz[0]
    sizey = base_sz[1]

    gstore = get_store(store)
    trains = gstore["train"]["inputs"]
    new_store = h5py.File("gray_seq.h5", "w")
    ins = new_store.create_dataset("inputs", shape=(trains.shape[0], sizex*sizey*seq_len), dtype=np.uint8)
    tmp = np.zeros(sizex*sizey*seq_len)
    for i in xrange(trains.shape[0]):
        dirctn = np.random.randint(0, 4)

        if (dirctn == 0):
            x = np.random.randint(0, 32-sizex)
            y = np.random.randint(0, 10)
            dy = np.random.randint(1, delta)
            dx = 0
        elif (dirctn == 1):
            x = np.random.randint(0, 10)
            y = np.random.randint(0, 32-sizey)
            dy = 0
            dx = np.random.randint(1, delta)
        elif (dirctn == 2):
            x = np.random.randint(0, 10)
            y = np.random.randint(0, 10)
            dy = np.random.randint(1, delta)
            dx = 0
        elif (dirctn == 3):
            x = np.random.randint(0, 10)
            y = np.random.randint(11, 21)
            dy = -np.random.randint(1, delta)
            dx = np.random.randint(1, delta)
        patch = trains[i].reshape(32, 32)
        for j in xrange(seq_len):
            tmp[sizex*sizey*j:sizex*sizey*(j+1)] = patch[x:x+sizex, y:y+sizey].ravel()
            x += dx
            y += dy
        ins[i, :] = tmp
    gstore.close()
    new_store.close()


def build_selected_store(store, selection, size, gray):
    """Build a hdf5 store with selected subbatches.
    """
    print "Writing to", store
    h5file = h5py.File(store, "w")

    _create_grp(store=h5file, grp_name="selection", batches=selection, gray=gray, size=size)

    print "Closing", store
    h5file.close()


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


def floatify_store(store, chunk=512, cache=False, exclude=[None], verbose=True):
    """A new store that contains stationary images from _store_.
    """
    if verbose:
        print "Floatify store", store
    sfn = store.filename.split(".")[0]
    name = hashlib.sha1(sfn + "float")
    name = name.hexdigest()[:8] + ".float.h5"
    if cache is True and exists(name):
        print "Using cached version ", name
        return h5py.File(name, 'r+')

    print "No cache, writing to", name
    flt = h5py.File(name, 'w')
    helpers.simply_float(store, flt, chunk=chunk, exclude=exclude)
    flt.attrs["Floatify"] = "from " + str(store.filename)
    return flt


def at_store(store, M, chunk=512, cache=False, exclude=[None], verbose=True):
    """
    """
    if verbose:
        print "AT store", store
    sfn = store.filename.split(".")[0]
    name = hashlib.sha1(sfn + str(M) + str(chunk))
    name = name.hexdigest()[:8] + ".at.h5"
    if cache is True and exists(name):
        if verbose: print "Using cached version ", name
        return h5py.File(name, 'r+')

    print "No cache, writing to", name
    at = h5py.File(name, 'w')
    helpers.at(store, at, M, chunk=chunk, exclude=exclude)
    at.attrs["AT"] = "from " + str(store.filename)
    return at


def _create_grp(store, grp_name, batches, gray=False, size=None):
    """
    """
    print "Creating", grp_name, "set."
    grp = store.create_group(grp_name)

    if size is None:
        dx, dy = 32, 32
    else:
        dx, dy = size[0], size[1]
    grp.attrs["patch_shape"] = (dx, dy)

    # color images, three channels -> dx*dy*3 input dimension
    color = 1 if gray else 3

    ins = grp.create_dataset("inputs", shape=(len(batches)*_batch_size, dx*dy*color), dtype=np.uint8)
    ins.attrs["patch_shape"] = (dx, dy)

    tars = grp.create_dataset("targets", shape=(len(batches)*_batch_size,))

    for i, batch in enumerate(batches):
        print "Reading batch from", join(_default_path, _batch_path, batch)
        dic = _unpickle(join(_default_path, _batch_path, batch))
        if gray:
            soon_gray = _pil_array(dic["data"], _batch_size)
            for j, p in enumerate(soon_gray):
                tmp = img.fromarray(p).convert("L")
                if size is not None:
                    tmp = tmp.resize(size, img.ANTIALIAS)
                ins[i*_batch_size+j] = np.asarray(tmp).ravel()
        else:
            ins[i*_batch_size:(i+1)*_batch_size] = dic["data"].ravel()
        tars[i*_batch_size:(i+1)*_batch_size] = dic["labels"]


def _unpickle(file):
    """
    Returns dictionaries in a 'data_batch' file.
    From http://www.cs.toronto.edu/~kriz/cifar.html
    """
    fo = open(file, 'rb')
    dic = cPickle.load(fo)
    fo.close()
    return dic


def _pil_array(pics, batch_size):
    """
    Given images from a batch in an array
    _pics_, reshape array such that
    PIL can handle it.
    """
    pics.resize(batch_size, 3, 32, 32)
    pics = pics.swapaxes(1,2)
    pics = pics.swapaxes(2,3)
    return pics


if __name__=="__main__":
    print "Building color store cifar10_32x32."
    build_store()
