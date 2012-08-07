"""
Handle data from notMNIST,

http://yaroslavvb.blogspot.com/2011/09/notmnist-dataset.html
"""


import h5py
from os import listdir
from os.path import dirname, join

import numpy as np
try:
    import Image as img
except:
    import PIL as img


from helpers import helpers


_default_path = dirname(__file__)
_default_name = join(_default_path, "not-mnist_28x28.h5")
_test_path = "notMNIST_small"
_train_path = "notMNIST_large"
_labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
_shape = (28,28)


def get_store(fname=_default_name):
    print "Loading from store", fname
    return h5py.File(fname, 'r')


def build_store(store=_default_name):
    """Build a hdf5 data store for not-MNIST
    """
    print "Writing to", store
    print "Please be patient."
    h5file = h5py.File(store, "w")

    _create_grp(store=h5file, grp_name="train", path=_train_path)
    _create_grp(store=h5file, grp_name="test", path=_test_path)

    print "Closing", store
    h5file.close()


def _create_grp(store, grp_name, path):
    print "Building store", grp_name
    _count = 0
    files = []
    for label in _labels:
        for name in listdir(join(path, label)):
            _count += 1
            # try/except because some of the png's are empty
            if name.endswith("png"):
                try:
                    glyph = img.open(join(path, label, name))
                    files.append(join(path, label, name))
                except:
                    print "Disregarding", join(path, label, name)                    

    count = len(files)
    print "Total files", _count
    print "Actually used", count

    print "Shuffle."
    helpers.shuffle_list(files)

    print "Filling store" , grp_name
    grp = store.create_group(grp_name)
    ins = grp.create_dataset("inputs", shape=(count, _shape[0]*_shape[1]), dtype=np.float32)
    tars = grp.create_dataset("targets", shape=(count, ), dtype=np.int)
    for i, f in enumerate(files):
        label = str.split(f, "/")[-2]
        try:
            glyph = img.open(f)
            ins[i] = np.asarray(glyph, dtype=np.float32).ravel()/255.
            tars[i] = int(ord(label) - ord('A'))
        except:
            pass


def invert(store, chunk=512):
    """1-pixelvalues.
    """
    print "Inverting ", store
    inv = h5py.File("inverted_not-mnist_28x28.h5", 'w')
    helpers.binary_invert(store, inv, chunk, exclude=["targets"])
    return inv


if __name__=="__main__":
    build_store()
