"""
"""


import numpy as np


def devisive(store, chunk=512, eps=1e-8):
    """Devisive normalization inplace.

    _store_ is build out of rows
    """
    for i in xrange(0, store.shape[0], chunk):
        norm = np.sqrt(np.sum(store[i:i+chunk]**2, axis=1) + eps)
        store[i:i+chunk] /= np.atleast_2d(norm).T
