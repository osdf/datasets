"""
Helper functions that are useful for
various kinds of datasets.
"""


import numpy as np
import scipy.linalg as la
import h5py
from time import strftime


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
    for key in store.keys():
        if type(store[key]) is h5py.Group:
            stationary(store[key], chunk=chunk, eps=eps, C=C)
        if type(store[key]) is h5py.Dataset:
            print "Stationary on ", key
            _stationary(store[key], chunk=chunk, eps=eps, C=C)


def _stationary(store, chunk=512, eps=1e-8, C=1.):
    """Subtract row-mean and divide by row-norm.

    _store_ has to be an np.array. Works __inplace__.
    """
    for i in xrange(0, store.shape[0], chunk):
        means = np.mean(store[i:i+chunk], axis=1)
        store[i:i+chunk] -= np.atleast_2d(means).T
        norm = np.sqrt(np.sum(store[i:i+chunk]**2, axis=1) + eps)
        store[i:i+chunk] /= np.atleast_2d(norm).T
        store[i:i+chunk] *= C


def shuffle(store):
    """Shuffle rows inplace.
    """
    for key in store.keys():
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


def visualize(array, rsz, rows, xtiles=None, fill=0):
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
    shape_r = rows
    shape_c = int(rsz/rows)
    
    # take care of extra pixels for borders
    pixelsy = ytiles * shape_r + ytiles + 1
    pixelsx = xtiles * shape_c + xtiles + 1
    # the tiling has this shape and _fill_ background
    tiling = fill*np.ones((pixelsy, pixelsx), dtype = 'uint8')
    
    for row in xrange(ytiles):
        for col in xrange(xtiles):
            if (col+row*xtiles) < fields.shape[0]:
                tile = fields[col + row * xtiles].reshape(shape_r, shape_c)
                tile = _scale_01(tile) * 255
                tiling[shape_r * row + row + 1:shape_r * (row+1) + row + 1, shape_c * col + col + 1:shape_c * (col+1) + col + 1] = tile
    return img.fromarray(tiling)


def pca(patches, covered=None, whiten=False, **schedule):
    """
    Assume _already_ normalized patches.
    """
    n, d = patches.shape
    # working with covariance + (svd on cov.) is
    # much faster than svd on patches directly.
    cov = np.dot(patches.T, patches)/n
    u, s, v = la.svd(cov, full_matrices=False)
    if covered is None:
        retained = d
    else:
        total = np.cumsum(s)[-1]
        retained = sum(np.cumsum(s/total) <= covered)
    print covered, whiten
    s = s[0:retained]
    u = u[:,0:retained]
    if whiten:
        comp = np.dot(u, np.diag(1./np.sqrt(s)))
    else:
        comp = u
    rotated = np.dot(patches, comp)
    return rotated, comp, s


def zca(patches, eps=0.1, **schedule):
    """
    Compute ZCA.
    """
    n, d = patches.shape
    cov = np.dot(patches.T, patches)/n
    u, s, v = la.svd(cov, full_matrices=False)
    print sum(s<eps)
    comp = np.dot(np.dot(u, np.diag(1./np.sqrt(s + eps))), u.T)
    zca = np.dot(patches, comp)
    return zca, comp, s


def unwhiten(X, comp):
    """
    Inverse process of whitening.
    _comp_ is assumed to be column wise.
    """
    uw = la.pinv2(comp)
    return np.dot(X, uw)


def apply_to_store(store, new_store, method, pars):
    """Apply _method_ to  images in _store_,
    save in _new_store_. _pars_ are parameters for
    _method_.
    """
    for key in store.keys():
        if type(store[key]) is h5py.Group:
            grp = new_store.create_group(name=key)
            apply_to_store(store[key], grp, method, pars)
        if type(store[key]) is h5py.Dataset:
            method(store, key, new_store, pars)


def crop(store, newst, x, y, dx, dy):
    """Generate a new store _newst_ from _store_ by
    cropping its images around at (x-dx, y-dy, x+dx, y+dy).
    _newst_ is simply an open, empty hdf5 file.
    """
    box = (x-dx, y-dy, x+dx, y+dy)
    apply_to_store(store, newst, _crop, box)
    return newst


def simply_float(store):
    """Just dump store into a new store
    that is (i) of dtype float and
    (ii) writeable.
    """
    tmp = ".".join([strftime("%Y-%m-%d-%H:%M:%S"), "float"])
    float_store = h5py.File(tmp, "w")
    apply_to_store(store, float_store, _floatify, 0)
    print "Temporary float store. Take care of", tmp
    return float_store


def _resize(store, key, new, shape):
    """
    """
    dx = shape[0]
    dy = shape[1]
    dset = new.create_dataset(name=key, shape=(store[key].shape[0], dx*dy), dtype=store[key].dtype)

    ps = store[key].attrs["patch_shape"]
    for i, p in enumerate(store[key]):
        p.resize(ps)
        resized_patch = img.fromarray(p).resize(shape, img.ANTIALIAS)
        dset[i] = np.asarray(resized_patch, dtype=dset.dtype).ravel()

    for attrs in store[key].attrs:
        dset.attrs[attrs] = store[key].attrs[attrs]
    dset.attrs["patch_shape"] = shape


def _crop(store, key, new, box):
    """
    """
    dx = box[2] - box[0]
    dy = box[3] - box[1]

    dset = new.create_dataset(name=key, shape=(store[key].shape[0], dx*dy), dtype=store[key].dtype)

    ps = store[key].attrs["patch_shape"]
    for i, p in enumerate(store[key]):
        p.resize(ps)
        croped_patch = img.fromarray(p).crop(box)
        dset[i] = np.asarray(croped_patch, dtype=dset.dtype).ravel()

    for attrs in store[key].attrs:
        dset.attrs[attrs] = store[key].attrs[attrs]
    dset.attrs["patch_shape"] = (dx, dy)


def _floatify(store, key, float_store, to_ignore):
    """Just dump store into a new store
    that is (i) of dtype float and
    (ii) writeable.
    """
    dset = float_store.create_dataset(name=key, shape=store[key].shape, dtype=np.float)

    for i, p in enumerate(store[key]):
        dset[i] = np.asarray(p, dtype=np.float)

    for attrs in store[key].attrs:
        dset.attrs[attrs] = store[key].attrs[attrs]
