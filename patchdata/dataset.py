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
try:
    import Image as img
except:
    import PIL as img

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


def get_store(fname="patchdata_64x64.h5", path=_default_path):
    print "Loading from store", fname
    return h5py.File(join(path, fname), 'r')


def build_store(fname="patchdata_64x64.h5", path=_default_path, dataset=dataset):
    print "Writing to", fname
    f = h5py.File(join(path, fname), "w")
    for ds in dataset:
        ds_path = join(path, ds)
        totals = _available_patches(ds_path)
        dset = f.create_dataset(name=ds, shape=(totals, patch_x*patch_y), dtype = np.uint8)
        bmps = totals / per_bmp
        print "For", ds, "reading a total of", totals, "patches from", bmps, "files into", fname
        for i in xrange(bmps):
            bmp = join(ds_path, ''.join(["patches", str(i).zfill(4), ".bmp"]))
            dset[i*per_bmp:(i+1)*per_bmp] = _crop_to_numpy(bmp)
        if totals%per_bmp > 0:
            bmp = join(ds_path, ''.join(["patches", str(bmps).zfill(4), ".bmp"])) 
            dset[-(totals%per_bmp):] = _crop_to_numpy(bmp)[:totals%per_bmp]
    f.attrs["dataset"] = dataset
    f.attrs["patch_shape"] = (patch_x, patch_y)
    print "Wrote", dataset, "to", fname, f


def build_evaluate_store(store, pair_list=_default_pairings, path=_default_path):
    """Put matching/non-matching pairs into a hdf5.

    There is one hdf5 per dataset, with the available number of pairs 
    in _pair_list_. These numbers form the groups of every store. 
    The store is using the original patches, scaled versions of 
    these patches should be generated with resize_store. 
    
    Every group has two datasets, 'match' and 'non-match',
    both arrays of equal length, the original pairs are formed by
    blocks of consecutive rows.

    Note: Every dataset in this store has dtype=np.float64, to ease
    later evaluation (avoid costly rebuilding everyting into float64).
    For this reason, reconsider when 'extending' pair_list into the 100.000
    range.
    """
    for ds in dataset:
        fname = "".join(["evaluate_", ds, "_", "64x64.h5"])
        print "\nBuilding evaluate store", fname, "for", pair_list
        f = h5py.File(join(path, fname), "w")
        for pairs in pair_list:
            grp = f.create_group(name=str(pairs))
            mtch, non_mtch, ids = matches(ds, pairs, path=path)
            _build_pairing_store(group=grp, name="match", pairings=mtch, store=store[ds])
            _build_pairing_store(group=grp, name="non-match", pairings=non_mtch, store=store[ds])
        f.attrs["dataset"] = ds
        f.attrs["patch_shape"] = (patch_x, patch_y)
        f.attrs["pairs"] = pair_list
        f.close()


def resize_store(store, new_store, size):
    """Resize images in _store_ to _size_, save in _new_store_.
    """
    for attrs in store.attrs:
        new_store[attrs] = store[attrs]
        if attrs is "patch_shape":
            new_store["patch_shape"] = size

    for key in store.keys():
        if type(store[key]) is h5py.Group:
            grp = new_store.create_group(name=key)
            resize_store(store[key], grp, size)
        if type(store[key]) is h5py.Dataset:
            dset = new_store.create_dataset(name=key, shape=(store[key].shape[0], size[0]*size[1]), dtype=store[key].dtype)
            dset.attrs["patch_size"] = size
            _resize(store[key], dset, size)


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


def select(store, dataset=dataset, index_set=[512, 512, 512], chunk=512, name="inputs", cache=True):
    """Select from the _dataset_s in _store_ some patches, specified by _index_set_.

    A _store_ has the three subsets ["liberty", "notredame", "yosemite"].
    Using _dataset_, specify from which subset(s) patches are selected.
    Default is to use all three subsets. _index_set_ defines for every
    subset, which patches (indentified by a list of ints) should be choosen.
    If _index_set_ is a list of numbers, then for every number a list of
    random numbers of this lenght is generated and used as indices instead.
    _chunk_ controls, how many patches are transfered in one got from 
    the _store_ to the newly generated dataset, a flat hdf5 dataset. 
    This dataset is named _name_.
    """
    random.seed()

    assert len(dataset) == len(index_set), "Every dataset needs its indices"

    train_size = 0
    valid_size = 0
    for i, j in index_set:
        train_size += i if type(i) is int else len(i)
        valid_size += j if type(j) is int else len(j)

    name = hashlib.sha1(str(store.attrs["shape"]) + str(dataset) + str(index_set))
    name = name.hexdigest()[:8]
    if cache is True and exists(name+".cache"):
        return h5py.File(name+".cache", 'r')

    select = h5py.File(name+".cache", 'w')
    sx, sy = store.attrs["shape"]
    
    train = select.create_group("train")
    train = train.create_dataset(name="inputs", shape=(train_size, sx*sy), dtype=np.float64)
    valid = select.create_group("validation")
    valid = valid.create_dataset(name="inputs", shape=(valid_size, sx*sy), dtype=np.float64)

    jt = 0
    jv = 0
    for d, (rt, rv) in izip(dataset, index_set):
        if type(rv) is int:
            print store.keys(), d, store[d]
            randoms = random.sample(xrange(store[d].shape[0]), rt+rv)
            rt = randoms[-rv:]
            rv = randoms[:rv]
            rt.sort()
            rv.sort()
        jt = _fill_up(store[d], train, indices=rt, pos=jt, chunk=chunk)
        jv = _fill_up(store[d], valid, indices=rv, pos=jv, chunk=chunk)
    return select


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
    tiling = fill*np.ones((pixelsy, pixelsx), dtype=np.uint8)
    
    for row in xrange(ytiles):
        for col in xrange(xtiles):
            if (col+row*xtiles) < fields.shape[0]:
                tile = fields[col + row * xtiles].reshape(shape, shape)
                tile = np.asarray(_scale_01(tile) * 255, dtype=np.uint8)
                tiling[shape * row + row + 1:shape * (row+1) + row + 1, shape * col + col + 1:shape * (col+1) + col + 1] = tile
    return img.fromarray(tiling)


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
        path = join(_default, ds)
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


def _resize(old_patches, new_patches, size):
    old_size = old_patches.attrs["patch_size"]
    for i, p in enumerate(old_patches):
        p.resize(old_size)
        resized_patch = img.fromarray(p).resize(size, img.ANTIALIAS)
        new_patches[i] = np.asarray(resized_patch, dtype=new_patches.dtype).ravel()


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


def _scale_01(arr, eps=1e-10):
    """Scale arr between [0,1].

    Useful for gray images to be produced with PIL.
    Tries to avoid unnecessary contrast enhancement.
    """
    newarr = arr.copy()
    mn = newarr.min()
    newarr -= mn 
    mx = newarr.max()
    newarr *= 1.0/(mx + eps)
    return newarr


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
    for j, pair in enumerate(pairings):
        p1, p2 = _patches_from_pair(pair=pair, store=store)
        dset[2*j] = p1
        dset[2*j+1] = p2


def _patches_from_pair(pair, store):
    return store[pair[0],:], store[pair[1],:]


if __name__=="__main__":
    build_store()