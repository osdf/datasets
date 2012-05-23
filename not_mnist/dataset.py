"""
Handle data from notMNIST,

http://yaroslavvb.blogspot.com/2011/09/notmnist-dataset.html
"""


import h5py
import cPickle
from os import listdir
from os.path import dirname, join
import glob

import numpy as np
try:
    import Image as img
except:
    import PIL as img

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
    shuffle_list(files)

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
            

####
#### should go in general helper module 
####
def shuffle_list(lst):
    """Shuffle _lst_ 
    """
    n = len(lst)
    for i in xrange(n):
        interval = n - i
        idx = i + np.random.randint(interval)
        lst[i], lst[idx] = lst[idx], lst[i]


def _scale_01(arr, eps=1e-10):
    """Scale arr between [0,1].

    Useful for gray images to be produced with PIL.
    Does some contrast enhancement.
    """
    newarr = arr.copy()
    mn = newarr.min()
    newarr -= mn 
    mx = newarr.max()
    newarr *= 1.0/(mx + eps)
    return newarr


def visualize(array, rsz, xtiles=None, fill=0):
    """Visualize flattened bitmaps.

    _array_ is supposed to be a 1d array that
    holds the bitmaps (of size _rsz_ each)
    sequentially. _rsz_ must be a square number.

    Specifiy the number of rows with _xtiles_.
    If not specified, the layout is approximately
    square. _fill_ defines the pixel border between
    patches (default is black (==0)).
    """
    sz = array.size
    fields = array.reshape(sz/rsz, rsz)
    
    # tiles per axes
    xtiles = xtiles if xtiles else int(np.sqrt(sz/rsz))
    ytiles = int(np.ceil(sz/rsz/(1.*xtiles)))
    shape = int(np.sqrt(rsz))
    
    # take care of extra pixels for borders
    pixelsy = ytiles * shape + ytiles + 1
    pixelsx = xtiles * shape + xtiles + 1
    # the tiling has this shape and _fill_ background
    tiling = fill*np.ones((pixelsy, pixelsx), dtype = 'uint8')
    
    for row in xrange(ytiles):
        for col in xrange(xtiles):
            if (col+row*xtiles) < fields.shape[0]:
                tile = fields[col + row * xtiles].reshape(shape, shape)
                tile = _scale_01(tile) * 255
                tiling[shape * row + row + 1:shape * (row+1) + row + 1, shape * col + col + 1:shape * (col+1) + col + 1] = tile
    return img.fromarray(tiling)


if __name__=="__main__":
    build_store()
