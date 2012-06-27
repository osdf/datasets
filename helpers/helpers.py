"""
"""


import numpy as np
import h5py

import numpy as np
try:
    import Image as img
except:
    import PIL as img


def stationary(store, chunk=512, eps=1e-8, C=1.):
    """Subtract mean and divide by norm.

    Works for input data that is stationary
    (the statistics of every input dimension
    follows the same distribution), e.g. image
    patches.
    """
    for key in store.keys:
        if type(store[key]) is h5py.Group:
            stationary(store[key], chunck=chunk, eps=eps, C=C)
        if type(store[key]) is h5py.Dataset:
            print "Stationary on ", key
            _stationary(store[key], chunk=chunk, eps=eps, C=C)


def _stationary(store, chunk=512, eps=1e-8, C=1.):
    """Subtract row-mean and divide by row-norm.

    _store_ has to be an np.array. Works __inplace__.
    """
    for i in xrange(0, store.shape[0], chunk):
        means = np.mean(store[i:i+chunk], axis=1)
        store[i:i+chunk] -= means
        norm = np.sqrt(np.sum(store[i:i+chunk]**2, axis=1) + eps)
        store[i:i+chunk] /= np.atleast_2d(norm).T
        store[i:i+chunk] *= C


def shuffle(store):
    """Shuffle rows inplace.
    """
    for key in store.keys:
        if type(store[key]) is h5py.Group:
            shuffle(store[key])
        if type(store[key]) is h5py.Dataset:
            print "Shuffle ", key
            _shuffle(store[key])


def _shuffle(store):
    """Shuffle rows inplace.
    _store_ has to an np.array.
    """
    N, _ = store.shape
    for i in xrange(N):
        interval = N - i
        idx = i + np.random.randint(interval)
        tmp = store[idx].copy()
        store[idx] = store[i].copy()
        store[i] = tmp


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
