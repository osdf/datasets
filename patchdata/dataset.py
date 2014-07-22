"""
This module provides handling
of the basic aspects of the patchdata set:
    - building hdf5 stores (for the original sized patches)
      in particular the stores to evaluate descriptors
    - resizing patches (works only on available stores)
    - randomly selecting patches
    - printing general information
"""


import subprocess
import random
from itertools import izip, product
from collections import defaultdict
from os.path import dirname, join, exists
import hashlib
import numpy as np
import h5py
import scipy.ndimage as simg


try:
    import Image as img
except:
    import PIL as img


from helpers import helpers


# the dataset comprises 3 subdatasets
dataset = ["liberty", "notredame", "yosemite"]

# size of an original patch in pixels
patch_x = 64
patch_y = 64

# patches come in bmp files, rows/cols of patches per file
rows = 16
cols = 16
per_bmp = rows*cols

# helpful defaults
_default_path = dirname(__file__)
_default_pairings = (500, 1000, 2500, 5000, 10000, 25000)


def get_store(fname="patchdata_64x64.h5", path=_default_path, verbose=True, access='r'):
    if verbose:
        print "Loading from store", fname
    return h5py.File(join(path, fname), access)


def build_store(fname="patchdata_64x64.h5", path=_default_path, dataset=dataset):
    print "Writing to", fname
    f = h5py.File(join(path, fname), "w")
    for ds in dataset:
        ds_path = join(path, ds)
        totals = _available_patches(ds_path)
        dset = f.create_dataset(name=ds, shape=(totals, patch_x*patch_y), dtype = np.uint8)
        bmps, mod = divmod(totals, per_bmp)
        print "For", ds, "reading a total of", totals, "patches from", bmps, "files into", fname
        for i in xrange(bmps):
            bmp = join(ds_path, ''.join(["patches", str(i).zfill(4), ".bmp"]))
            dset[i*per_bmp:(i+1)*per_bmp] = _crop_to_numpy(bmp)
        if mod > 0:
            bmp = join(ds_path, ''.join(["patches", str(bmps).zfill(4), ".bmp"])) 
            dset[-(totals%per_bmp):] = _crop_to_numpy(bmp)[:mod]
    f.attrs["dataset"] = dataset
    f.attrs["patch_shape"] = (patch_y, patch_x)
    f.close()
    print "Wrote", dataset, "to", fname, f


def build_evaluate_store(store, dataset=dataset, pair_list=_default_pairings, path=_default_path, tag=None):
    """Put matching/non-matching pairs into a hdf5.

    There is one hdf5 per dataset, with the available number of pairs 
    in _pair_list_. These numbers form the groups of every store. 
    The store is using the original patches, scaled versions of 
    these patches should be generated with resize.
    
    Every group has two datasets, 'match' and 'non-match',
    both arrays of equal length, the original pairs are formed by
    blocks of consecutive rows.

    Note: Every dataset in this store has dtype=np.float64, to ease
    later evaluation (avoid costly rebuilding everyting into float64).
    For this reason, reconsider when 'extending' pair_list into the 100.000
    range.
    """
    if tag is None:
        tag = ""
    else:
        tag = "".join([tag, "_"])
    for ds in dataset:
        fname = "".join([tag, "evaluate_", ds, "_", "64x64.h5"])
        print "\nBuilding evaluate store", fname, "for", pair_list
        f = h5py.File(join(path, fname), "w")
        for pairs in pair_list:
            grp = f.create_group(name=str(pairs))
            mtch, non_mtch, ids = matches(ds, pairs, path=path)
            _build_pairing_store(group=grp, name="match", pairings=mtch, store=store[ds])
            _build_pairing_store(group=grp, name="non-match", pairings=non_mtch, store=store[ds])
        f.attrs["dataset"] = ds
        f.attrs["patch_shape"] = (patch_y, patch_x)
        f.attrs["pairs"] = pair_list
        f.close()


def build_supervised_store(dataset=dataset, sz=250000, pairings=True):
    """Helper function for building a (flat) store for supervised training
    on 'dataset'. It has _sz_ many pairs of matches and _sz_ many pairs
    of non-matches. If _pairings_ is True, first build the split match/non-match
    base hdf5 for _dataset_.
    """
    print "Building supervised flat store for", dataset, sz

    if pairings:
        print "Building non-flat base store first."
        store = get_store()
        tag = "supervised_{0}".format(sz)
        if type(dataset) is str:
            dataset = [dataset]
        build_evaluate_store(store, dataset=dataset, pair_list=[sz], tag=tag)
        store.close()

    fsd = []
    for ds in dataset:
        fname = "supervised_{0}_evaluate_{1}_64x64.h5".format(sz, ds)
        # fuse this store
        store = get_store(fname)
        fname = "supervised_{0}_evaluate_{1}_fuse_64x64.h5".format(sz, ds)
        fused = fuse_store(store, str(sz), fname=fname)
        fsd.append(fused)
    print "Remember to close stores in returned list!", fsd
    print "Probably remove unnececssary intermediate stores."
    return fsd


def build_supervised_scale_store(dataset, sz, scale="laplace", depth=3, fused=None):
    """Helper function for building a (flat) store for supervised training
    on 'dataset' with scale information. It has _sz_ many pairs of matches and _sz_ many pairs
    of non-matches.
    """
    print "Building store for scale {0}, with {1} scales".format(scale, depth)

    if fused is None:
        print "Need to build scaled store first. This takes time ..."
        fused = build_supervised_store(dataset, sz)

    if type(fused) is not list:
        fused = [fused]
    
    scaled = []
    for fsd in fused:
        scale_st = pyramid_store(fsd, schema=scale, params=[depth], 
                exclude=['targets'])
        scaled.append(scale_st)
    print "Remember to close stores in returned list!", scaled
    return scaled


def info(dataset=dataset):
    """Print out basic information about _dataset_.

    _dataset_ has to be an array/tuple. default
    is ["liberty", "yosemite", "notredame"].
    """
    summary = summarize(dataset)
    for ds in summary.keys():
        patchset = summary[ds]
        print "\nPatchset:", ds
        print "No. of patches:", patchset["entries"]
        id_count = patchset["counts"]
        mn = min(id_count.keys())
        mx = max(id_count.keys())
        print "Min/Max of Patches per 3Did:", (len(id_count[mn]), mn), (len(id_count[mx]), mx)
        count_list = [(len(id_count[k]), k) for k in id_count.keys()]
        count_list.sort(reverse=True)
        print "Counts:", count_list[0:5]
        print "No. of different 3Dids:", sum((tpl[0] for tpl in count_list))


def select(store, dataset=dataset, index_set=[(512, 32), (512, 32), (512, 32)],
        chunk=512, cache=True, dim=patch_x*patch_y):
    """Select from the _dataset_s in _store_ some patches, specified by _index_set_.

    A _store_ has the three subsets ["liberty", "notredame", "yosemite"].
    Using _dataset_, specify from which subset(s) patches are selected.
    Default is to use all three subsets. _index_set_ defines for every
    subset, which patches (indentified by a list of ints) should be choosen.
    If _index_set_ is a list of numbers, then for every number a list of
    random numbers of this length is generated and used as indices instead.
    Note that the final _store_ has a group 'training' and a group
    'validation', thus _index_set_ needs a tuple for every subset given
    in _dataset_. _chunk_ controls, how many patches are transfered in one go from 
    the _store_ to the newly generated dataset, a flat hdf5 dataset. 
    """
    random.seed()

    assert len(dataset) == len(index_set), "Every dataset needs its indices"

    name = hashlib.sha1(str(store.attrs["patch_shape"]) + str(dataset) + str(index_set))
    name = name.hexdigest()[:8] + ".select.h5"
    if cache is True and exists(name):
        print "Using cached version ", name
        return h5py.File(name, 'r+')

    select = h5py.File(name, 'w')
    print "No cache, writing to", name
    select.attrs["patch_shape"] = store.attrs["patch_shape"]
    train_size = 0
    valid_size = 0
    for i, j in index_set:
        train_size += i if type(i) is int else len(i)
        valid_size += j if type(j) is int else len(j)
   
    train = select.create_group("train")
    train.attrs["patch_shape"] = store.attrs["patch_shape"]
    
    train = train.create_dataset(name="inputs", shape=(train_size, dim), dtype=np.float64)
    train.attrs["patch_shape"] = store.attrs["patch_shape"]
    
    valid = select.create_group("validation")
    valid.attrs["patch_shape"] = store.attrs["patch_shape"]
    
    valid = valid.create_dataset(name="inputs", shape=(valid_size, dim), dtype=np.float64)
    valid.attrs["patch_shape"] = store.attrs["patch_shape"]

    jt = 0
    jv = 0
    for d, (rt, rv) in izip(dataset, index_set):
        if type(rv) is int:
            print "Producing randomized selection", d, rt, rv, store[d]
            randoms = random.sample(xrange(store[d].shape[0]), rt+rv)
            rt = randoms[:-rv]
            rv = randoms[-rv:]
            rt.sort()
            rv.sort()
        jt = _fill_up(store[d], train, indices=rt, pos=jt, chunk=chunk)
        jv = _fill_up(store[d], valid, indices=rv, pos=jv, chunk=chunk)
    select.attrs["Original"] = "from " + str(store.filename)
    helpers.shuffle(select)
    return select


def matches(dataset, pairs, path=_default_path):
    """Return _pairs_ many match/non-match pairs for _dataset_.

    _dataset_ is one of "liberty", "yosemite", "notredame".
    Every dataset has a number of match-files, that 
    have _pairs_ many matches and non-matches (always
    the same number).

    The naming of these files is confusing, e.g. if there are 500 matching
    pairs and 500 non-matching pairs the file name is
    'm50_1000_1000_0.txt' -- in total 1000 patch-ids are used for matches,
    and 1000 patch-ids for non-matches. These patch-ids need not be
    unique.
    
    Also returns the used patch ids in a list.
    """
    match_file = ''.join(["m50_", str(2*pairs), "_", str(2*pairs), "_0.txt"]) 
    match_file = join(path, dataset, match_file)
    
    print pairs, "pairs each (matching/non_matching) from", match_file
    
    return pairings(match_file)


def pairings(fname):
    """Extract all matching and non matching pairs from _fname_.

    Every line in the matchfile looks like:
        patchID1 3DpointID1 unused1 patchID2 3DpointID2 unused2
    'matches' have the same 3DpointID. 
    
    Every file has the same number of matches and non-matches.
    """
    match_file = open(fname)
    # collect patches (id), and match/non-match pairs
    patch_ids, match, non_match = [], [], []
    
    for line in match_file:
        match_info = line.split()
        p1_id, p1_3d, p2_id, p2_3d = int(match_info[0]), int(match_info[1]), int(match_info[3]), int(match_info[4])
        if p1_3d == p2_3d:
            match.append((p1_id, p2_id))
        else:
            non_match.append((p1_id, p2_id))
        patch_ids.append(p1_id)
        patch_ids.append(p2_id)
    
    patch_ids = list(set(patch_ids))
    patch_ids.sort()
    
    assert len(match) == len(non_match), "Different number of matches and non-matches."
    
    return match, non_match, patch_ids


def summarize(dataset):
    """Get basic info from datasets in list _dataset_.
    """
    summary = dict()
    for ds in dataset:
        summary[ds] = dict()
        # 3D ids are in info.txt, one line per 64x64 patch
        path = join(_default_path, ds)
        info = open(join(path, "info.txt"))
        # collect all 3D id counts in a dictionary:
        # keys of this dictionary are how many counts a 3D id has,
        # to each count we list the 3D ids
        id_count = defaultdict(list)
        # id's are numbered from 0 on per dataset.
        count, current_id = 0, 0
        for i, line in enumerate(info):
            id_3d = int(line.split(' ')[0])
            if current_id != id_3d:
                id_count[count].append(current_id)
                count, current_id = 0, id_3d
            else:
                count = count + 1
        summary[ds]["counts"] = id_count
        summary[ds]["entries"] = i+1
    return summary


def unpair_store(store, grp, tag="match", cache=False):
    """
    Given a store with match/nonmatch pairs, extract the members 
    of the pairs from _grp_|tag as two seperat stores.
    """
    print "Unpairing store", store, ", group ", grp, tag
    sfn = store.filename.split(".")[0]
    name = hashlib.sha1(sfn + str(grp))

    name1 = name.hexdigest()[:8] + "." + tag + ".ins.h5"
    name2 = name.hexdigest()[:8] + "." + tag + ".outs.h5"
    if cache is True and exists(name1) and exists(name2):
        print "Using cached version ", name1, name2
        return h5py.File(name1, 'r+'), h5py.File(name2, 'r+')

    print "No cache, writing to", name1, name2
    p1 = h5py.File(name1, 'w')
    p2 = h5py.File(name2, 'w')

    p1.attrs["Unpaired"] = "from " + tag + " " + str(store.filename)
    p2.attrs["Unpaired"] = "from " + tag + " " + str(store.filename)

    group = store[grp][tag]
    n, d = group.shape
    n = n/2
    p1g = p1.create_dataset(name="inputs", shape=(n, d), dtype=group.dtype)
    p2g = p2.create_dataset(name="inputs", shape=(n, d), dtype=group.dtype)

    for j in xrange(n):
        p1g[j] = group[2*j]
        p2g[j] = group[2*j + 1]
    return p1, p2


def crop_store(store, x, y, dx, dy, cache=False, verbose=True):
    """A new store that contains cropped images from _store_.
    _x_, _y_, _dx_ and _dy_ are the cropping parameters.
    """
    if verbose:
        print "Cropping from store", store, "with (x,y);(dx, dy)", x, y, dx, dy
    sfn = store.filename.split(".")[0]
    name = hashlib.sha1(sfn + str(x) + str(y) + str(dx) + str(dy))

    name = name.hexdigest()[:8] + ".crop.h5"
    if cache is True and exists(name):
        if verbose:
            print "Using cached version ", name
        return h5py.File(name, 'r+')

    print "No cache, writing to", name
    crop = h5py.File(name, 'w')
    helpers.crop(store, crop, x, y, dx, dy)
    crop.attrs["Cropped"] = "from " + str(store.filename)
    return crop


def stationary_store(store, eps=1e-8, C=1., div=1., chunk=512, cache=False,
        exclude=[None], verbose=True, fname=None):
    """A new store that contains stationary images from _store_.
    """
    if verbose:
        print "Stationarize store", store, "with eps, C, div" , eps, C, div, fname
    if fname is None:
        sfn = store.filename.split(".")[0]
        name = hashlib.sha1(sfn + str(C) + str(eps) + str(chunk))
        name = name.hexdigest()[:8] + ".stat.h5"
    else:
        name = fname
    if cache is True and exists(name):
        print "Using cached version ", name
        return h5py.File(name, 'r+')

    print "No cache, writing to", name
    stat = h5py.File(name, 'w')
    helpers.stationary(store, stat, chunk=chunk, eps=eps, C=C, div=div, exclude=exclude)
    stat.attrs["Stationary"] = "from " + str(store.filename)
    return stat


def fuse_store(store, key, groups=["match", "non-match"], labels=[1,0], 
        stride=2, cache=False, fname=None):
    """
    A new store that fuses images from 'match'/'non-match' parts
    into one main store.
    """
    print "Fuse store", store, ", key", str(key)
    if fname is None:
        sfn = store.filename.split(".")[0]
        name = hashlib.sha1(sfn + str(groups) + str(stride))
        name = name.hexdigest()[:8] + ".fuse.h5"
    else:
        name = fname
    if cache is True and exists(name):
        print "Using cached version ", name
        return h5py.File(name, 'r+')

    print "No cache, writing to", name
    fuse = h5py.File(name, 'w')
    helpers.fuse(store[key], fuse, groups=groups, labels=labels, stride=stride)
    fuse.attrs["Fused"] = "from " + str(store.filename) + ", " + str(groups)
    return fuse


def merge_store(store1, store2, stride=4, cache=False):
    """
    A new store that merges images from store1 and store2
    into one main store.
    """
    print "Merge stores", store1, store2
    sfn1 = store1.filename.split(".")[0]
    sfn2 = store2.filename.split(".")[0]
    name = hashlib.sha1(sfn1 + sfn2 + str(stride))
    name = name.hexdigest()[:8] + ".merge.h5"
    if cache is True and exists(name):
        print "Using cached version ", name
        return h5py.File(name, 'r+')

    print "No cache, writing to", name
    merge = h5py.File(name, 'w')
    helpers.merge(store1, store2, merge, stride=stride)
    merge.attrs["Merged"] = "from " + sfn1 + ", " + sfn2
    return merge


def concat_store(store, group, chunk=512, cache=False, exclude=[None], verbose=True):
    """A new store that contains stationary images from _store_.
    """
    if verbose:
        print "Concatenate store", store, "with group", group 
    sfn = store.filename.split(".")[0]
    name = hashlib.sha1(sfn + str(group) + str(chunk))
    name = name.hexdigest()[:8] + ".concat.h5"
    if cache is True and exists(name):
        print "Using cached version ", name
        return h5py.File(name, 'r+')

    print "No cache, writing to", name
    conc = h5py.File(name, 'w')
    helpers.concat(store, conc, chunk=chunk, grp=group)
    conc.attrs["Concatenated"] = "from " + str(store.filename)
    return conc


def row0_store(store, chunk=512, cache=False, verbose=True):
    """A new store that contains 0-mean images from _store_.
    """
    if verbose:
        print "Row0 store", store
    sfn = store.filename.split(".")[0]
    name = hashlib.sha1(sfn + str(chunk))
    name = name.hexdigest()[:8] + ".row0.h5"
    if cache is True and exists(name):
        if verbose:
            print "Using cached version ", name
        return h5py.File(name, 'r+')

    print "No cache, writing to", name
    r0 = h5py.File(name, 'w')
    helpers.row0(store, r0, chunk=chunk)
    r0.attrs["Row0"] = "from " + str(store.filename)
    return r0 


def feat0_store(store, to_sub, chunk=512, exclude=[None], cache=False):
    """A new store that is featurewise 0-mean.
    """
    print "Feat0 store", store
    sfn = store.filename.split(".")[0]
    name = hashlib.sha1(sfn + str(to_sub) + str("feat0_store") + str(chunk))
    name = name.hexdigest()[:8] + ".feat0.h5"
    if cache is True and exists(name):
        print "Using cached version ", name
        return h5py.File(name, 'r+')

    print "No cache, writing to", name
    f0 = h5py.File(name, 'w')
    helpers.feat_sub(store, f0, chunk=chunk, sub=to_sub, exclude=exclude)
    f0.attrs["Feat0"] = "from " + str(store.filename)
    return f0 


def feat_std1_store(store, to_div, chunk=512, cache=False):
    """
    New store has standard deviation 1 per feature.
    """
    print "Feat_std1 store", store
    sfn = store.filename.split(".")[0]
    name = hashlib.sha1(sfn + str(to_div) + str("feat_std1_store") + str(chunk))
    name = name.hexdigest()[:8] + ".feat_std1.h5"
    if cache is True and exists(name):
        print "Using cached version ", name
        return h5py.File(name, 'r+')

    print "No cache, writing to", name
    fstd1 = h5py.File(name, 'w')
    helpers.feat_div(store, fstd1, chunk=chunk, div=to_div)
    fstd1.attrs["Feat_std1"] = "from " + str(store.filename)
    return fstd1


def gstd1_store(store, to_div, chunk=512, cache=False, verbose=True):
    """A new store that has global std = 1.
    """
    if verbose:
        print "GStd1 store", store
    sfn = store.filename.split(".")[0]
    name = hashlib.sha1(sfn + str(to_div) + str("gstd1_store") + str(chunk))
    name = name.hexdigest()[:8] + ".gstd1.h5"
    if cache is True and exists(name):
        if verbose: print "Using cached version ", name
        return h5py.File(name, 'r+')

    print "No cache, writing to", name
    std = h5py.File(name, 'w')
    helpers.global_div(store, std, chunk=chunk, div=to_div)
    std.attrs["GStd1"] = "from " + str(store.filename)
    return std


def double_store(store, chunk=512, cache=False, exclude=[None], verbose=True):
    """A new store that contains stationary images from _store_.
    """
    if verbose:
        print "Double store", store, "excluding ", exclude
    sfn = store.filename.split(".")[0]
    name = hashlib.sha1(sfn + str(exclude) + str(chunk))
    name = name.hexdigest()[:8] + ".double.h5"
    if cache is True and exists(name):
        print "Using cached version ", name
        return h5py.File(name, 'r+')

    print "No cache, writing to", name
    stat = h5py.File(name, 'w')
    helpers.double(store, stat, chunk=chunk, exclude=exclude)
    stat.attrs["Double"] = "from " + str(store.filename)
    return stat


def resize_store(store, shape, cache=False, exclude=[None], verbose=True):
    """A new store that contains resized images from _store_.
    """
    if verbose:
        print "Resizing store", store, "to new shape", shape
    sfn = store.filename.split(".")[0]
    name = hashlib.sha1(sfn + str(shape))
    name = name.hexdigest()[:8] + ".resz.h5"
    if cache is True and exists(name):
        if verbose: print "Using cached version ", name
        return h5py.File(name, 'r+')

    print "No cache, writing to", name
    rsz = h5py.File(name, 'w')
    helpers.resize(store, rsz, shape, exclude=exclude)
    rsz.attrs["Resized"] = "from " + str(store.filename)
    return rsz


def fward_store(store, fward, D, chunk=512, cache=False, exclude=[None], verbose=True):
    """
    """
    if verbose:
        print "fward store", store
    sfn = store.filename.split(".")[0]
    name = hashlib.sha1(sfn + str(fward) + str(chunk))
    name = name.hexdigest()[:8] + ".fw.h5"
    if cache is True and exists(name):
        if verbose:
            print "Using cached version ", name
        return h5py.File(name, 'r+')

    print "No cache, writing to", name
    fw = h5py.File(name, 'w')
    helpers.fward(store, fw, fward, D, chunk=chunk)
    fw.attrs["FWARD"] = "from " + str(store.filename) + " " + str(fward)
    return fw


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


def zeroone_store(store, chunk=512, cache=False):
    """
    """
    print "Zeroone store", store
    sfn = store.filename.split(".")[0]
    name = hashlib.sha1(sfn+str(chunk))
    name = name.hexdigest()[:8] + ".zo.h5"

    if cache is True and exists(name):
        print "Using cached version ", name
        return h5py.File(name, 'r+')

    print "No cache, writing to", name
    zo = h5py.File(name, 'w')
    helpers.zeroone(store, zo, chunk=chunk)
    zo.attrs["ZO"] = "from " + str(store.filename)
    return zo 


def zeroone_group(store, chunk=512, group=["match", "non-match"], cache=False):
    """
    """
    print "Zeroone Group", store, group
    sfn = store.filename.split(".")[0]
    name = hashlib.sha1(sfn+str(chunk)+"zeroone_group"+str(group))
    name = name.hexdigest()[:8] + ".zogrp.h5"

    if cache is True and exists(name):
        print "Using cached version ", name
        return h5py.File(name, 'r+')

    print "No cache, writing to", name
    zo = h5py.File(name, 'w')
    helpers.zeroone_group(store, zo, group=group, chunk=chunk)
    zo.attrs["ZO_GRP"] = "from " + str(store.filename)
    return zo 


def pyramid_store(store, schema="laplace", params=[3], chunk=512, cache=False, 
        exclude=[None], verbose=True, fname=None):
    """A new store that contains pyramid images from _store_.
    """
    if verbose:
        print "Pyramid store", store, "with params (depth is first!), schema:", params, params[0], schema
    
    if schema == "laplace":
        ending = ".lapy.h5"
    elif schema == "lcn":
        ending = ".lcnpy.h5"
    elif schema == "fovea":
        ending = ".fovpy.h5"
    else:
        assert False, "Unkown pyramid schema %s"%schema

    if fname is None:
        sfn = store.filename.split(".")[0]
        name = hashlib.sha1(sfn + str(schema) + str(params) + str(chunk))
        name = name.hexdigest()[:8] + ending
    else:
        name = fname
    if cache is True and exists(name):
        print "Using cached version ", name
        return h5py.File(name, 'r+')

    print "No cache, writing to", name
    pyr = h5py.File(name, 'w')
    helpers.pyramid(store, pyr, chunk=chunk, schema=schema, params=params, exclude=exclude)
    pyr.attrs["Pyramid"] = "from " + str(store.filename)
    return pyr


def pyramidfuse_store(store, schema="laplace", depth=(), chunk=512, cache=False, exclude=[None], verbose=True):
    """A new store that contains pyramid images from _store_.
    """
    if verbose:
        print "Pyramid Fuse store", store, "with depth, schema:", depth, schema
    sfn = store.filename.split(".")[0]
    name = hashlib.sha1(sfn + str(schema) + str(depth) + str(chunk))
    name = name.hexdigest()[:8] + ".lapyfuse.h5"
    if cache is True and exists(name):
        print "Using cached version ", name
        return h5py.File(name, 'r+')

    print "No cache, writing to", name
    pyr = h5py.File(name, 'w')
    helpers.pyramid_fuse(store, pyr, chunk=chunk, schema=schema, depth=depth, exclude=exclude)
    pyr.attrs["PyramidFused"] = "from " + str(store.filename)
    return pyr


def _crop_to_numpy(patchfile, ravel=True):
    """Convert _patchfile_ to a numpy array with patches per row.

    A _patchfile_ is a .bmp file.
    """
    patches = img.open(patchfile)
    arr = [] 
    for row, col in product(xrange(rows), xrange(cols)):
        ptch = patches.crop((col*patch_x, row*patch_y, (col+1)*patch_x, (row+1)*patch_y))
        if ravel:
            arr.append(np.array(ptch).ravel())
        else:
            arr.append(np.array(ptch))
    return np.array(arr)


def _fill_up(from_store, to_store, indices, pos, chunk):
    """Fill _to_store_ with patches from _from_store_.
    """
    for i in xrange(0, len(indices), chunk):
        _idx = indices[i:i+chunk]
        to_store[pos:pos+len(_idx)] = from_store[_idx]
        pos += len(_idx)
    return pos


def _available_patches(dataset):
    """Number of patches in _dataset_ (a path).
    
    Only available through the line count 
    in info.txt -- use unix 'wc'.
    
    _path_ is supposed to be a path to
    a directory with bmp patchsets.
    """
    fname = join(dataset, "info.txt")
    p = subprocess.Popen(['wc', '-l', fname],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    result, err = p.communicate()
    if p.returncode != 0:
        raise IOError(err)
    return int(result.strip().split()[0])


def _build_pairing_store(group, name, pairings, store):
    """Create a dataset with pairs from _pairings_.
    
    The dataset _name_ is below group _group_.  _pairings are 
    pairs of indices that refer to _store_. The new store 
    (a hdf5 dataset) is a flat array, yet, blocks of 
    two consecutive rows form a pair.

    Note: The new store has dtype=np.float64.
    """
    dset = group.create_dataset(name=name, shape=(2*len(pairings), patch_x*patch_y), dtype=np.float64)
    dset.attrs["patch_shape"] = (patch_y, patch_x)
 
    for j, pair in enumerate(pairings):
        p1, p2 = _patches_from_pair(pair=pair, store=store)
        dset[2*j] = p1
        dset[2*j+1] = p2


def _patches_from_pair(pair, store):
    return store[pair[0],:], store[pair[1],:]


def build_phantom_store(store, dataset, index_set, nmbrs,
        deltax, deltay, scale, rot, dimx, dimy,
        path=_default_path, nonmatch=False, tag=None):
    """
    """
    print "Building phantom store out of {0}".format(dataset)
    total = 0
    for idx in index_set:
        total = total+idx[0]
    total = total*nmbrs

    if tag is None:
        tag = ""
    else:
        tag = "".join([tag, "_"])
    fname = "".join(["phantom_", tag, str(total), '-', str(nmbrs), "_", "{0}x{1}.h5".format(dimx, dimy)])
    f = h5py.File(join(path, fname), "w")
    print "Writing phantom data to", f

    sel = select(store, dataset=dataset, index_set=index_set, cache=False)
    grp = f.create_group(name="train")
    grp.attrs["patch_shape"] = sel.attrs["patch_shape"]

    if nonmatch:
        print "Positive and negative pairs -- dataset is named 'match'."
        mtch_name = "match"
    else:
        print "Only positive pairs -- dataset is named 'inputs'."
        mtch_name = "inputs"

    # positive pair set is built always
    shape = sel['train']['inputs'].shape
    train = grp.create_dataset(name=mtch_name, shape=(2*shape[0]*nmbrs, dimx*dimy),
            dtype=np.float32)
    i = 0
    for elem in sel['train']['inputs']:
        x = elem.reshape(patch_x, patch_y)
        ph = phantom(x, nmbrs, deltax, deltay, scale, rot, dimx, dimy)
        # get inner original patch of x
        ox, oy = x.shape[0]//2, x.shape[1]//2
        xmin, xmax = ox - dimx//2, ox + dimx//2
        ymin, ymax = oy - dimy//2, oy + dimy//2
        tmp = x[xmin:xmax, ymin:ymax]
        tmp = tmp.ravel()
        for j, p in enumerate(ph):
            train[i+2*j, :] = tmp
            train[i+2*j+1,:] = p
        i = i + 2*nmbrs
    helpers._shuffle_pairs(train)
    sel.close()
    
    if nonmatch:
        print "Building negative pairs with a new selection store."
        sel = select(store, dataset=dataset, index_set=index_set, cache=False)
        shape = sel['train']['inputs'].shape
        train = grp.create_dataset(name="non-match", shape=(shape[0]*nmbrs, dimx*dimy),
                dtype=np.float32)
        _nmbrs = int(sqrt(nmbrs))
        assert _nmbrs**2 == nmbrs, "For negative matches, need a valid number of pairings."

        i = 0
        for x in sel['train']['inputs']:
            x = elem.reshape(patch_x, patch_y)
            ph1 = phantom(x, _nmbrs, deltax, deltay, scale, rot=180, dimx=dimx, dimy=dimy)
            rnds = np.random.randint(0, shape[0], size=_nmbrs)
            for j, p in enumerate(ph1):
                for k, r in enumerate(rnds):
                    tmp = sel['train']['inputs'][r]
                    tmp = tmp.reshape(patch_x, patch_y)
                    tmp = phantom(tmp, 1, deltax, deltay, scale, rot=180, dimx=dimx, dimy=dimy)
                    train[i+2*(j*_nmbrs + k),:] = p
                    train[i+2*(j*_nmbrs + k)+1,:] = tmp[0]
            i = i + 2*nmbrs
        helpers._shuffle_pairs(train)
        sel.close()
    f.close()


def phantom(x, nmbrs, deltax, deltay, scale, rot, dimx, dimy):
    """Generate _phantoms_ many phantom images of _x_
    """
    phantoms = np.zeros((nmbrs, dimx*dimy))
    for i in xrange(nmbrs):
        dx = np.random.randint(-deltax, deltax+1)
        dy = np.random.randint(-deltay, deltay+1)
        sc = np.random.randint(-scale, scale+1)
        rt = np.random.randint(-rot, rot+1)

        pic = simg.shift(x, (dy, dx), mode='reflect')
        pic = simg.rotate(pic, rt, mode='reflect')
        shape = pic.shape
        ox, oy = shape[0]//2, shape[1]//2
        xmin, xmax = ox - 16 + sc//2, ox + 16 - sc//2
        ymin, ymax = oy - 16 + sc//2, oy + 16 - sc//2
        pic = pic[xmin:xmax, ymin:ymax]
        pic = simg.zoom(pic, (1.0*dimx)/(xmax-xmin))
        phantoms[i, :] = pic.ravel()
    return phantoms


def flip_patches(patches):
    """
    Flip patches (either 90 degrees left right, or 180 degrees).
    Assume that patches come in pairs.
    """
    n, d = patches.shape
    dx = int(np.sqrt(d))
    tmp = patches.reshape((n, dx, dx))
    result = np.zeros((n, d))
    for j in xrange(n/2):
        flips = np.random.rand()
        # only flip at most one image
        if flips < 0.33:
            # no flipping
            result[2*j, :] = patches[2*j]
            result[2*j + 1, :] = patches[2*j + 1]
        else:
            # either p1 or p2 flips
            p2 = img.fromarray(tmp[2*j + 1])
            # determine random flip
            rnd1 = np.random.rand()
            if rnd1 < 0.33:
                flip = img.ROTATE_90
            elif rnd1 < 0.66:
                flip = img.ROTATE_180
            else:
                flip = img.ROTATE_270

            if flips < 0.66:
                # p1 flips
                p1 = img.fromarray(tmp[2*j])
                p1 = p1.transpose(flip)
                result[2*j, :] = np.asarray(p1).ravel()
                result[2*j+1,:] = patches[2*j+1]
            else:
                p2 = img.fromarray(tmp[2*j+1])
                p2 = p2.transpose(flip)
                result[2*j, :] = patches[2*j]
                result[2*j+1,:] = np.asarray(p2).ravel()
    return result


def gauss_patches(patches, sigma=0.1):
    """
    Noise one of the two patches with gaussian noise.
    This could be done more efficiently (matrix * matrix),
    but I want to make sure that only one of two patches
    gets noised at most. With this constraint, the for loop
    seems to be the most straight forward.
    """
    n, d = patches.shape
    result = np.zeros((n, d))
    for j in xrange(n/2):
        gaussian = np.random.rand()
        # only noise at most one image
        if gaussian < 0.33:
            # no noise
            result[2*j, :] = patches[2*j]
            result[2*j+1, :] = patches[2*j + 1]
        elif gaussian < 0.66:
            result[2*j, :] = patches[2*j] + np.random.normal(scale=sigma, size=(d,))
            result[2*j+1, :] = patches[2*j+1]
        else:
            result[2*j, :] = patches[2*j]
            result[2*j+1, :] = patches[2*j+1] + np.random.normal(scale=sigma, size=(d,))
    return result


def snp_patches(patches, drop):
    """
    Bernoulli noise (salt'n pepper) on patches.
    """
    n, d = patches.shape
    result = np.zeros((n, d))
    for j in xrange(n/2):
        noise = np.random.rand()
        # only noise at most one image
        if noise < 0.33:
            # no noise
            result[2*j, :] = patches[2*j]
            result[2*j+1, :] = patches[2*j + 1]
        elif noise < 0.66:
            result[2*j, :] = patches[2*j] * (np.random.uniform(size=(d,)) > drop)
            result[2*j+1, :] = patches[2*j+1]
        else:
            result[2*j, :] = patches[2*j]
            result[2*j+1, :] = patches[2*j+1] * (np.random.uniform(size=(d,)) > drop)
    return result


if __name__=="__main__":
    build_store()
