"""
"""


import numpy as np
from itertools import product


def cosine_dist(v1, v2):
    """Cosine similarity between two vectors, v1 and v2."""
    n1 = np.sqrt(np.sum(v1**2))
    n2 = np.sqrt(np.sum(v2**2))
    return 1 - np.dot(v1, v2)/(n1*n2)


def l2_dist(v1, v2):
    """L2 distance between v1 and v2."""
    return np.sqrt(np.sum((v1-v2)**2))


def ham_dist(v1, v2):
    """Hamming distance between v1 and v2."""
    return np.sum(v1 != v2)


def l1_dist(v1, v2):
    """L1 distance between v1 and v2."""
    dist = np.sum(np.abs(v1-v2)) 
    return dist


_dist_table = {
    "L2": l2_dist
    ,"L1": l1_dist
    ,"COSINE":cosine_dist
    ,"HAMMING": ham_dist
}


_full_dist = ["L2", "L1", "COSINE", "HAMMING"]
_cont_dist = ["L2", "L1", "COSINE"]


def id(v):
    """v1 is not normalized."""
    return v


def l2(v):
    """v is l2 normalized."""
    return v/np.sqrt(np.sum(v**2))


def l1(v):
    """v is l1 normalized."""
    return v/np.sum(np.abs(v))


def binary(v):
    """Binarize v.

    Assumes that v is [0,1]^n
    """
    return v>0.5


_norm_table = {
    "id": id
    ,"l2": l2
    ,"l1": l1
    ,"01": binary
}


_full_norms = ["id", "l2", "l1", "01"]
_cont_norms = ["id", "l2", "l1"]


def roc(matches, non_matches):
    """ROC for distances in _matches_ and _non_matches_.
    """
    matches = np.array(matches)
    non_matches = np.array(non_matches)
    # number of true positives/false positives
    total_tp = float(len(matches))
    total_fp = float(len(non_matches))
    # Threshold finding: I want to find tp and fp. Therefore
    # look a distances between median, 75%Quartil (q3) and
    # maximum distance in the matching histogramm.
    med = np.median(matches)
    q3 = np.median(matches[matches > med])
    mx = np.max(matches)
    # compute threshold by linear interplating 
    # between median, q3 and max distance
    thresholds = list(np.linspace(q3, mx, 200))
    # summary: list of tuples, threshold and (tp,fp) pair.
    curve = []
    for thresh in thresholds:
        tp = np.sum(matches < thresh)/total_tp
        fp = np.sum(non_matches < thresh)/total_fp
        curve.append({"true_positive":tp, "false_positive":fp, "threshold": thresh})
    return curve


def fp_at_95(curve):
    """
    Get false positive rate at 95% tp.
    """
    rates = [elem["false_positive"] for elem in curve if elem["true_positive"] >= 0.945]
    if len(rates) == 0:
        rates.append(1)
    return rates[0]


def evaluate(eval_set, distances=_cont_dist, normalizations=_cont_norms,
        latent=_nop):
    """
    """
    rocs = dict()

    for pairs in eval_set:
        roc_pair = dict()
        dset = eval_set[pairs]
        matches = latent(dset["match"])
        non_matches = latent(dset["non-match"])
        for dist, norm in product(distances, normalizations):
            if dist is "HAM" and norm is not "01":
                continue
            m_dist = _histogram(matches, int(pairs), _dist_table[dist], _norm_table[norm])
            nonm_dist = _histogram(non_matches, int(pairs), _dist_table[dist], _norm_table[norm])
            curve = roc(m_dist, nonm_dist)
            fp95 = fp_at_95(curve)
            print "Evaluate:", pairs, dist, norm, fp95
            roc_pair[(dist, norm)] = {"fp_at_95": fp95, "roc": curve, 
                    "m_dist": m_dist, "nonm_dist": nonm_dist}
        rocs[pairs] = roc_pair
    return rocs


def _histogram(dataset, pairs, dist, norm):
    """Compute distance histogram.
    """
    hist = []
    for i in xrange(pairs):
        v1, v2 = dataset[2*i], dataset[2*i+1]
        hist.append(dist(norm(v1), norm(v2)))
    return hist


def _nop(x):
    return x
