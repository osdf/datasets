"""
"""


import numpy as np
from itertools import product

SMALL = 1e-8

def cosine_dist(v1, v2):
    """Cosine similarity between two vectors, v1 and v2."""
    n1 = np.sqrt(np.sum(v1**2) + SMALL)
    n2 = np.sqrt(np.sum(v2**2) + SMALL)
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


def chi_dist(v1, v2):
    """Chi-squared histogramm distance.
    """
    t1 = (v1-v2)**2
    t2 = v1 + v2 + 1e-6
    dist = 0.5 * np.sum(t1/t2)
    return np.sqrt(dist)


def jsd(v1, v2):
    """
    Jensen Shannon divergence.
    """
    sumv = (v1+v2)/2.
    t1 = v1*np.log((v1/(sumv+SMALL))+SMALL) + (1-v1)*np.log( (1-v1)/(1-sumv+SMALL)+SMALL)
    t2 = v2*np.log(v2/(sumv+SMALL)+SMALL) + (1-v2)*np.log( (1-v2)/(1-sumv+SMALL)+SMALL)
    return np.sqrt(np.abs(0.5*np.sum(t1) + 0.5*np.sum(t2)))


_dist_table = {
    "L2": l2_dist
    ,"L1": l1_dist
    ,"COSINE":cosine_dist
    ,"HAMMING": ham_dist
    ,"CHI": chi_dist
    ,"JSD": jsd
}


_full_dist = ["L2", "L1", "COSINE", "HAMMING", "JSD"]
_cont_dist = ["L2", "L1", "JSD"]


def id(v):
    """v1 is not normalized."""
    return v


def l2(v):
    """v is l2 normalized."""
    return v/np.sqrt(np.sum(v**2) + SMALL)


def l1(v):
    """v is l1 normalized."""
    return v/(np.sum(np.abs(v)) + SMALL)


def binary(v):
    """Binarize v.

    Assumes that v is [0,1]^n
    """
    return v>0.5

def sqrt(v):
    """Sqrt-ing the vector.
    Heuristic to make L2 distance
    norm work _occasionally_ better.
    """
    return np.sqrt(v)

#def binar(v, t=0.11):
#"""'bin' und 'L1' produce good results, 48% for yosemite
#    bv = 1*(v>t)
#    return l2(bv)
#    check for 0.0045 again!
#def binar(v, t=0.00544): #best setting for 81, trained on notredame (yose: 46%)
#    #n = l2(v)
#    n=v# good for 0.03!!
#    #if v.shape[0] == 64:
#    #    t = np.array([0.017, 0.08, 0.218, 0.219, 0.117, 0.137, 0.042, 0.0021, 0.090, 0.178, 0.196, 0.0618, 0.0389, 0.0196, 0.0058, 0.090, 0.0128, 0.091, 0.155, 0.088, 0.0517, 0.0119, 0.123, 0.129, 0.0361, 0.0126, 0.081, 0.229, 0.0449, 0.110, 0.072, 0.129, 0.163, 0.037, 0.0434, 0.028, 0.051, 0.0489, 0.0016, 0.0616, 0.0927, 0.021, 0.1, 0.065, 0.0781, 0.1688, 0.0146, 0.0159, 0.198, 0.026, 0.0052, 0.0277, 0.0074, 0.0115, 0.051, 0.00248, 0.038, 0.081, 0.0026, 0.1124, 0.0965, 0.0933, 0.019, 0.0107])/10
#    bv = 1*(n>t)
#    return bv
def binarize(v, thresh, **kwargs):
    return 1*(v>thresh)


_norm_table = {
    "id": id
    ,"l2": l2
    ,"l1": l1
    ,"01": binary
    ,"sqrt": sqrt
    ,"bin": binarize
}


_full_norms = ["id", "l2", "l1", "01", "sqrt"]
_cont_norms = ["id", "l2", "l1", "sqrt"]


def roc(matches, non_matches):
    """ROC for distances in _matches_ and _non_matches_.
    """
    sortedm = matches[:]
    sortedm.sort()
    dist_at_95 = sortedm[int(0.95*len(matches))]
    dist_at_75 = sortedm[int(0.75*len(matches))]
    dist_max = sortedm[-1]
    thresholds = list(np.linspace(dist_at_75, dist_max, 200))
    matches = np.array(matches)
    non_matches = np.array(non_matches)
    # number of true positives/false positives
    total_tp = float(len(matches))
    total_fp = float(len(non_matches))
    ## Threshold finding: I want to find tp and fp. Therefore
    ## look a distances between median, 75%Quartil (q3) and
    ## maximum distance in the matching histogramm.
    #med = np.median(matches)
    #q3 = np.median(matches[matches > med])
    #mx = np.max(matches)
    ## compute threshold by linear interplating
    ## between median, q3 and max distance
    #thresholds = list(np.linspace(q3, mx, 200))
    ## summary: list of tuples, threshold and (tp,fp) pair.
    curve = [{"true_positive": 0.95, "false_positive": np.sum(non_matches < dist_at_95)/total_fp, "threshold": dist_at_95}]
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


def _nop(x):
    """
    """
    return x


def evaluate(eval_set, distances=_cont_dist,
        normalizations=_cont_norms, latent=_nop, verbose=True):
    """
    """
    print "Evaluate", eval_set.attrs['dataset']
    if verbose:
        for att in eval_set.attrs:
            if att != "dataset":
                    print att, ":", eval_set.attrs[att]

    rocs = dict()

    for pairs in eval_set:
        roc_pair = dict()
        dset = eval_set[pairs]
        matches = latent(dset["match"])
        non_matches = latent(dset["non-match"])
        for dist, norm in product(distances, normalizations):
            if np.logical_xor(dist is "HAMMING", norm is "01"):
                continue
            m_dist = _dhistogram(matches, int(pairs), _dist_table[dist], _norm_table[norm])
            nonm_dist = _dhistogram(non_matches, int(pairs), _dist_table[dist], _norm_table[norm])
            curve = roc(m_dist, nonm_dist)
            fp95 = fp_at_95(curve)
            
            if verbose:
                print pairs, dist, norm, fp95
            
            roc_pair[(dist, norm)] = {"fp_at_95": fp95, "roc": curve, 
                    "m_dist": m_dist, "nonm_dist": nonm_dist}
        rocs[pairs] = roc_pair
    return rocs


def _dhistogram(dataset, pairs, dist, norm):
    """Compute distance histogram.
    """
    hist = []
    for i in xrange(pairs):
        v1, v2 = dataset[2*i], dataset[2*i+1]
        hist.append(dist(norm(v1), norm(v2)))
    return hist


def _ahistogram(dataset, latent, norm='id'):
    """Build activation histogram.
    """
    hist = []
    for m in dataset:
        for inpt in dataset[m]:
            x = latent(inpt.reshape(1, -1))
            hist.extend(_norm_table[norm](x).ravel())
    return hist
