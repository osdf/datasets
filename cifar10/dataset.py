"""
Handle data from CIFAR 10.

"""


import h5py
import cPickle
from os.path import dirname, join

import numpy as np
try:
    import Image as img
except:
    import PIL as img

_default_path = dirname(__file__)
_default_name = join(_default_path, "cifar10_32x32.h5")
_default_gray = join(_default_path, "cifar10_gray_32x32.h5")
_batch_path = "cifar-10-batches-py"
_train = ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4"]
_valid = ["data_batch_5"]
_test  = ["test_batch"]
_batch_size = 10000


def get_store(fname=_default_name):
    print "Loading from store", fname
    return h5py.File(fname, 'r')


def build_store(store=_default_name):
    """Build a hdf5 data store for CIFAR.
    """
    print "Writing to", store
    h5file = h5py.File(store, "w")

    _create_grp(store=h5file, grp_name="train", batches=_train)
    _create_grp(store=h5file, grp_name="validation", batches=_valid)
    _create_grp(store=h5file, grp_name="test", batches=_test)

    print "Closing", store
    h5file.close()


def build_gray_store(store=_default_gray, size=None):
    """Build a hdf5 data store for CIFAR10, inputs are gray.
    """
    print "Writing to", store
    h5file = h5py.File(store, "w")

    _create_grp(store=h5file, grp_name="train", batches=_train, gray=True, size=size)
    _create_grp(store=h5file, grp_name="validation", batches=_valid, gray=True, size=size)
    _create_grp(store=h5file, grp_name="test", batches=_test, gray=True, size=size)

    print "Closing", store
    h5file.close()


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
