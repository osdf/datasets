"""
"""


import numpy as np
import h5py


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
        norm = np.sqrt(np.sum(store[i:i+chunk]**2, axis=1) + eps)
        store[i:i+chunk] /= np.atleast_2d(norm).T


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
        array[idx] = store[i].copy()
        array[i] = tmp
