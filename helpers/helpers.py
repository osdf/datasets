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


def shuffle(store):
    """Shuffle rows inplace.
    """
    if type(store) == h5py.File:
        print "Shuffling inplace ..."

    for key in store.keys():
        if type(store[key]) is h5py.Group:
            shuffle(store[key])
        if type(store[key]) is h5py.Dataset:
            print "...", store, key
            _shuffle(store[key])


def _shuffle(store):
    """Shuffle rows inplace.
    _store_ has to behave like
    an numpy.array.
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


def visualize(array, rsz, shape_r=None, xtiles=None, fill=0):
    """Visualize flattened bitmaps.

    _array_ is supposed to be a 1d array that
    holds the bitmaps (of size _rsz_ each)
    sequentially. _rsz_ must be a square number
    if _shape_r is None.

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
    if shape_r is None:
        shape_r = int(np.sqrt(rsz))
    shape_c = int(rsz/shape_r)
    
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


def hinton(array, sqr_sz = 9):
    """A hinton diagram without matplotlib.
    Code definetely has potential for improvement.

    _array_ is the one to visualize. _sqr_sz_ is
    the length of a square. Bigger -> more details.

    See https://gist.github.com/292018
    """
    dx, dy = array.shape
    W = 2**np.ceil(np.log(np.abs(array).max())/np.log(2))
    # take care of extra pixels for borders
    pixelsy = dy * sqr_sz + dy + 1
    pixelsx = dx * sqr_sz + dx + 1
    tiling = 128*np.ones((pixelsx, pixelsy), dtype = 'uint8')
    for (x,y), w in np.ndenumerate(array):
        xpos = x * sqr_sz + x + 1 + int(sqr_sz/2 + 1)
        ypos = y * sqr_sz + y + 1 + int(sqr_sz/2 + 1)
        dw = int(np.abs(w)/W * sqr_sz)/2 + 1
        cw = (w > 0) * 255
        tiling[xpos - dw:xpos + dw, ypos - dw:ypos+dw] = cw
    return img.fromarray(tiling)


def global_std(store, chunk=512):
    """Compute global standard deviation.
    Assumes that data is _overall_ zero mean.
    """
    N, d = store.shape
    var = 0
    for i in xrange(0, N, chunk):
        var += ((store[i:i+chunk])**2).sum()
    return np.sqrt(var/(1.*N*d))


def pca(data, covered=None, whiten=False, chunk=512, **schedule):
    """
    Assume _already_ normalized patches. This is supposed
    to work specifically for hdf5 _data_.
    """
    n, d = data.shape
    cov = np.zeros((d,d))
    for i in xrange(0, n, chunk):
        tmp = np.array(data[i:i+chunk])
        cov += np.dot(tmp.T, tmp)
    cov /= n 
    u, s, v = la.svd(cov, full_matrices=False)
    if covered is None:
        retained = d
    else:
        total = np.cumsum(s)[-1]
        retained = sum(np.cumsum(s/total) <= covered)
    s = s[0:retained]
    u = u[:,0:retained]
    if whiten:
        comp = np.dot(u, np.diag(1./np.sqrt(s)))
    else:
        comp = u
    return comp, s


def zca(data, eps=1e-5, chunk=512, **schedule):
    """
    Compute ZCA.
    """
    n, d = data.shape
    cov = np.zeros((d,d))
    for i in xrange(0, n, chunk):
        tmp = np.array(data[i:i+chunk])
        cov += np.dot(tmp.T, tmp)
    cov /= n 
    u, s, v = la.svd(cov, full_matrices=False)
    comp = np.dot(np.dot(u, np.diag(1./np.sqrt(s + eps))), u.T)
    return comp, s


def unwhiten(X, comp):
    """
    Inverse process of whitening.
    _comp_ is assumed to be column wise.
    """
    uw = la.pinv2(comp)
    return np.dot(X, uw)


def apply_to_group(store, new, method, pars, group):
    """
    """
    for key in store.keys():
        if key in group:
            method(store, key, new, pars)
            return
        else:
            if type(store[key]) is h5py.Group:
                grp = new.create_group(name=key)
                apply_to_group(store[key], grp, method, pars, group)
                for attrs in store.attrs.keys():
                    new.attrs[attrs] = store.attrs[attrs]
                for attrs in grp.attrs.keys():
                    new.attrs[attrs] = grp.attrs[attrs]
            elif type(store[key]) is h5py.Dataset:
                    clone_dataset(store, key, new)


def apply_to_store(store, new, method, pars, exclude=[None]):
    """Apply _method_ to images in _store_,
    save in _new_store_. _pars_ are parameters for
    _method_. If _method_ should not be applied to
    groups/datasets with certain keys, put these
    keys into _exclude_ (i.e. pass in as list of strings).
    """
    for key in store.keys():
        if type(store[key]) is h5py.Group:
            if key in exclude:
                continue
            grp = new.create_group(name=key)
            apply_to_store(store[key], grp, method, pars, exclude)
            for attrs in store.attrs.keys():
                new.attrs[attrs] = store.attrs[attrs]
            for attrs in grp.attrs.keys():
                new.attrs[attrs] = grp.attrs[attrs]

        if type(store[key]) is h5py.Dataset:
            if key in exclude:
                clone_dataset(store, key, new)
            else:
                method(store, key, new, pars)


def clone_group(store, key, new):
    """
    """
    grp = new.create_group(name=key)
    for k in store[key].keys():
        if type(store[key][k]) is h5py.Group:
            clone_group(store[key], k, grp)

        if type(store[key]) is h5py.Dataset:
            clone_dataset(store[key], k, grp)


def clone_dataset(store, key, new):
    """
    """
    shape = store[key].shape
    chunk = 512
    dset = new.create_dataset(name=key, shape=shape, dtype=store[key].dtype)
    for i in xrange(0, shape[0], chunk):
        dset[i:i+chunk] = store[key][i:i+chunk]

    for attrs in store[key].attrs:
        dset.attrs[attrs] = store[key].attrs[attrs]

    for attrs in store.attrs:
        new.attrs[attrs] = store.attrs[attrs]


def resize(store, new, shape, exclude=[None]):
    """Resize elements of _store_ to _shape_.
    """
    apply_to_store(store, new, _resize, shape, exclude=exclude)


def crop(store, new, x, y, dx, dy, exclude=[None]):
    """Generate a new store _new_ from _store_ by
    cropping its images around at (x-dx, y-dy, x+dx, y+dy).
    _new_ is simply an open, empty hdf5 file.
    """
    box = (x-dx, y-dy, x+dx, y+dy)
    apply_to_store(store, new, _crop, box, exclude=exclude)
    return new


def simply_float(store, new, chunk, exclude=[None]):
    """Just dump store into a new store
    that is (i) of dtype float and
    (ii) writeable.
    FIXME: Needs rework (get new store from caller)
    """
    apply_to_store(store, new, _floatify, pars=chunk, exclude=exclude)
    return new


def binary_invert(store, new, chunk, exclude=[None]):
    """
    """
    apply_to_store(store, new, _binary_inv, chunk, exclude=exclude)
    return new


def stationary(store, new, chunk=512, eps=1e-8, C=1., div=1., exclude=[None]):
    """Generate a new store _new_ from _store_ by
    'stationary' normalization of _store_.
    """
    pars = (chunk, eps, C, div)
    apply_to_store(store, new, _stationary, pars, exclude=exclude)


def pyramid(store, new, chunk=512, schema="Laplace", params=[3], exclude=[None]):
    """Generate a new store _new_ from _store_ by
    building a pyramid of type _schema_ with depth _depth_.
    """
    pars = (chunk, schema, params)
    apply_to_store(store, new, _pyramid, pars, exclude=exclude)


def pyramid_fuse(store, new, chunk=512, schema="Laplace", depth=(), exclude=[None]):
    """Generate a new store _new_ from _store_ by
    building a pyramid of type _schema_ with depth _depth_.
    """
    pars = (chunk, schema, depth)
    apply_to_store(store, new, _pyramid_fuse, pars, exclude=exclude)


def double(store, new, chunk=512, exclude=[None]):
    """Generate a new store _new_ from _store_ by
    doubling every entry.
    """
    pars = (chunk,)
    apply_to_store(store, new, _double, pars, exclude=exclude)


def row0(store, new, chunk=512, exclude=[None]):
    """Every row has mean 0.
    """
    pars = chunk
    apply_to_store(store, new, _row0, pars, exclude=exclude)


def at(store, new, M, chunk=512, exclude=[None]):
    """Affine Transformation
    """
    pars = (chunk, M)
    apply_to_store(store, new, _at, pars, exclude=exclude)


def fward(store, new, fward, D, chunk=512, exclude=[None]):
    """Affine Transformation
    """
    pars = (chunk, fward, D)
    apply_to_store(store, new, _fward, pars, exclude=exclude)


def zeroone(store, new, chunk=512, exclude=[None]):
    """0/1 normalization
    """
    pars = chunk
    apply_to_store(store, new, _zeroone, pars, exclude=exclude)


def zeroone_group(store, new, group, chunk=512):
    """
    """
    pars = chunk
    apply_to_group(store, new, _zeroone_group, pars, group)


def feat_mean(store, chunk=512):
    """Featurewise mean.
    """
    N, d = store.shape
    sm = 0
    for i in xrange(0, N, chunk):
        sm += store[i:i+chunk].sum(axis=0)
    return sm/(1.*N)


def feat_std(store, chunk=512):
    """
    Standard deviation of features (columns).
    Assumes that features have mean 0.
    """
    N, d = store.shape
    cdiv = 0.
    for i in xrange(0, N, chunk):
        cdiv += (store[i:i+chunk]**2).sum(axis=0)
    return np.sqrt(cdiv/(1.*N))


def fuse(store, new, groups, labels, stride=2, exclude=[None]):
    """
    Fuse all members in groups into one store _new_.
    Assumes that all groups have the same size!
    """
    assert len(groups) == len(labels)

    # Can not use apply_to_store because it works per subgroup
    n, d = store[groups[0]].shape
    k = len(groups)
    newg = new.create_group("train")
    inputs = newg.create_dataset(name="inputs", shape=(k*n, d), dtype=store[groups[0]].dtype)
    targets = newg.create_dataset(name="targets", shape=(k*n,), dtype=np.int)

    for i in xrange(n/stride):
        base = k*stride*i
        for j, g in enumerate(groups):
            inputs[base+j*stride:base+(j+1)*stride, :] = store[g][stride*i:stride*(i+1),:]
            targets[base+j*stride:base+(j+1)*stride] = labels[j]
    for attrs in store[groups[0]].attrs:
        inputs.attrs[attrs] = store[groups[0]].attrs[attrs]

def merge(store1, store2, new, stride=4, exclude=[None]):
    """
    """
    for key in store1.keys():
        if key in store2:
            if type(store1[key]) is h5py.Group:
                if key in exclude:
                    continue
                grp = new.create_group(name=key)
                merge(store1[key], store2[key], grp, stride, exclude)

            if type(store1[key]) is h5py.Dataset:
                if key in exclude:
                    continue
                else:
                    _mergeds(store1[key], store2[key], new, key, stride)


def feat_sub(store, new, chunk, sub, exclude=[None]):
    """
    """
    pars = (chunk, sub)
    apply_to_store(store, new, _feat_sub, pars, exclude=exclude)


def feat_div(store, new, chunk, div, exclude=[None]):
    """
    """
    pars = (chunk, div)
    apply_to_store(store, new, _feat_div, pars, exclude=exclude)


def global_div(store, new, chunk, div, exclude=[None]):
    """
    """
    pars = (chunk, div)
    apply_to_store(store, new, _global_div, pars, exclude=exclude)


def concat(store, new, chunk, grp):
    """
    Concatenate all datasets below group grp into one dataset ->
    substitute group grp by a dataset.
    """
    pars = (chunk,)
    apply_to_group(store, new, _concat, pars, grp)


def _binary_inv(store, key, new, chunk):
    """
    """
    shape = store[key].shape
    dset = new.create_dataset(name=key, shape=shape, dtype=store[key].dtype)
    for i in xrange(0, shape[0], chunk):
        dset[i:i+chunk] = 1 - store[key][i:i+chunk]

    for attrs in store[key].attrs:
        dset.attrs[attrs] = store[key].attrs[attrs]

    for attrs in store.attrs:
        new.attrs[attrs] = store.attrs[attrs]


def _feat_sub(store, key, new, pars):
    """Subtract featurewise.
    """
    chunk, sub = pars[0], pars[1]
    shape = store[key].shape
    dset = new.create_dataset(name=key, shape=shape, dtype=store[key].dtype)
    for i in xrange(0, shape[0], chunk):
        dset[i:i+chunk] = store[key][i:i+chunk] - sub

    for attrs in store[key].attrs:
        dset.attrs[attrs] = store[key].attrs[attrs]
    dset.attrs['feature mean'] = (sub.mean(), sub.std())

    for attrs in store.attrs:
        new.attrs[attrs] = store.attrs[attrs]
    new.attrs['feature mean'] = (sub.mean(), sub.std())


def _feat_div(store, key, new, pars):
    """Divide featurewise.
    """
    chunk, div = pars[0], pars[1]
    shape = store[key].shape
    dset = new.create_dataset(name=key, shape=shape, dtype=store[key].dtype)
    for i in xrange(0, shape[0], chunk):
        dset[i:i+chunk] = store[key][i:i+chunk]/div.T

    for attrs in store[key].attrs:
        dset.attrs[attrs] = store[key].attrs[attrs]
    dset.attrs['feature std'] = (div.mean(), div.std())

    for attrs in store.attrs:
        new.attrs[attrs] = store.attrs[attrs]
    new.attrs['feature std'] = (div.mean(), div.std())


def _global_div(store, key, new, pars):
    """
    """
    chunk, div = pars[0], pars[1]
    shape = store[key].shape
    dset = new.create_dataset(name=key, shape=shape, dtype=store[key].dtype)
    for i in xrange(0, shape[0], chunk):
        dset[i:i+chunk] = store[key][i:i+chunk]/div

    for attrs in store[key].attrs:
        dset.attrs[attrs] = store[key].attrs[attrs]
    dset.attrs['DIV'] = div

    for attrs in store.attrs:
        new.attrs[attrs] = store.attrs[attrs]
    new.attrs['DIV'] = div


def _zeroone(store, key, new, pars):
    """Zero/One normalization.
    """
    chunk = pars
    shape = store[key].shape 
    dset = new.create_dataset(name=key, shape=shape, dtype=store[key].dtype)
    mn = np.inf
    mx = -np.inf
    for i in xrange(0, shape[0], chunk):
        tmp_min = store[key][i:i+chunk][:].min()
        tmp_max = store[key][i:i+chunk][:].max()
        if tmp_min < mn:
            mn = tmp_min
        if tmp_max > mx:
            mx = tmp_max

    diff = mx-mn
    for i in xrange(0, shape[0], chunk):
        dset[i:i+chunk] = (store[key][i:i+chunk] - mn)/diff
 
    for attrs in store[key].attrs:
        dset.attrs[attrs] = store[key].attrs[attrs]
    dset.attrs['ZeroOne MinMax'] = (mn, mx)

    for attrs in store.attrs:
        new.attrs[attrs] = store.attrs[attrs]
    new.attrs['ZeroOne MinMax'] = (mn, mx)
    print key, (diff, mn, mx)


def _zeroone_group(store, new, pars):
    """Zero/One normalization on complete group
    of _store_. Assumes that store is
    a group of datasets!
    """
    chunk = pars
    mn = np.inf
    mx = -np.inf
    for key in store.keys():
        shape = store[key].shape
        for i in xrange(0, shape[0], chunk):
            tmp_min = store[key][i:i+chunk][:].min()
            tmp_max = store[key][i:i+chunk][:].max()
            if tmp_min < mn:
                mn = tmp_min
            if tmp_max > mx:
                mx = tmp_max

    diff = mx - mn

    for key in store.keys():
        dset = new.create_dataset(name=key, shape=shape, dtype=store[key].dtype)
        for i in xrange(0, shape[0], chunk):
            dset[i:i+chunk] = (store[key][i:i+chunk] - mn)/diff
        for attrs in store[key].attrs:
            dset.attrs[attrs] = store[key].attrs[attrs]
        dset.attrs['ZeroOne MinMax'] = (mn, mx)

    for attrs in store.attrs:
        new.attrs[attrs] = store.attrs[attrs]
        new.attrs['ZeroOne MinMax'] = (mn, mx)


def _at(store, key, new, pars):
    """Apply affine transformation (at) to 
    _store[key]_ members and build dataset in _new_.
    """
    chunk, M = pars[0], pars[1]
    n, _ = store[key].shape
    shape = (n, M.shape[1])
    dset = new.create_dataset(name=key, shape=shape, dtype=store[key].dtype)

    for i in xrange(0, n, chunk):
        dset[i:i+chunk] = np.dot(store[key][i:i+chunk], M)
 
    for attrs in store[key].attrs:
        dset.attrs[attrs] = store[key].attrs[attrs]
    dset.attrs['Affine shape'] = M.shape[1]

    for attrs in store.attrs:
        new.attrs[attrs] = store.attrs[attrs]
    new.attrs['Affine shape'] = M.shape[1]


def _fward(store, key, new, pars):
    """Apply fward function to 
    _store[key]_ members and build dataset in _new_.
    """
    chunk, fward, D = pars[0], pars[1], pars[2]
    n, _ = store[key].shape
    shape = (n, D)
    dset = new.create_dataset(name=key, shape=shape, dtype=store[key].dtype)

    for i in xrange(0, n, chunk):
        dset[i:i+chunk] = fward(store[key][i:i+chunk])
 
    for attrs in store[key].attrs:
        dset.attrs[attrs] = store[key].attrs[attrs]
    dset.attrs['fward'] = str(fward)

    for attrs in store.attrs:
        new.attrs[attrs] = store.attrs[attrs]
    new.attrs['fward'] = str(fward)


def _mergeds(store1, store2, new, name, stride):
    """
    """
    s1shape = store1.shape
    s2shape = store2.shape
    if len(s1shape) > 1:
        n1, d1 = s1shape
        n2, d2 = s2shape
        assert d1==d2 and n1==n2
        ds = new.create_dataset(name=name, shape=(n1+n2, d1), dtype=store1.dtype)
    else:
        n1 = s1shape[0]
        n2 = s2shape[0]
        ds = new.create_dataset(name=name, shape=(n1+n2,), dtype=store1.dtype)

    for i in xrange(n1/stride):
        base_stores = i*stride
        base = i*2*stride
        ds[base:base+stride] = store1[base_stores:base_stores+stride]
        ds[base+stride:base+2*stride] = store2[base_stores:base_stores+stride]


def _stationary(store, key, new, pars):
    """Subtract row-mean and divide by row-norm.

    _store_ has to be an np.array. Works __inplace__.
    """
    chunk, eps, C, div = pars[0], pars[1], pars[2], pars[3]
    
    dset = new.create_dataset(name=key, shape=store[key].shape, dtype=store[key].dtype)

    for i in xrange(0, store[key].shape[0], chunk):
        means = np.mean(store[key][i:i+chunk], axis=1)
        dset[i:i+chunk] = store[key][i:i+chunk] - np.atleast_2d(means).T
        norm = np.sqrt(np.sum(dset[i:i+chunk]**2, axis=1)/div + eps)
        dset[i:i+chunk] /= np.atleast_2d(norm).T
        dset[i:i+chunk] *= C

    for attrs in store[key].attrs:
        dset.attrs[attrs] = store[key].attrs[attrs]
    dset.attrs['StationaryC'] = C

    for attrs in store.attrs:
        new.attrs[attrs] = store.attrs[attrs]
    new.attrs['StationaryC'] = C


def _gaussian2d(width, sigma):
    """
    Gaussian filter of total width _width_.
    """
    d, r = divmod(width, 2)
    r = 1 if r==0 else 0
    const = -2*sigma**2
    grid = np.arange(-d + 0.5*r, d+1 - 0.5*r, 1)
    grid = np.exp((grid**2)/const)
    gridT = grid.reshape(grid.size, 1)
    result = gridT*grid
    return np.asarray(result/np.sum(result))


def _lcn_filters(fmaps, depth, width, sigma):
    """
    """
    import theano

    filters = np.zeros((fmaps, fmaps, width, width), dtype=theano.config.floatX)
    d2 = depth//2
    for i in xrange(fmaps):
        for j in xrange(i-d2, i+d2+1):
            if (j >= 0) and (j <fmaps):
                filters[i, j, :, :] = _gaussian2d(width, sigma)
        fi_sum = np.sum(filters[i])
        filters[i] /= fi_sum
    return filters


def _lcn(image, im_shape, fmaps, pool_depth, width, sigma):
    """
    """
    import theano
    import theano.tensor as T
    from theano.tensor.nnet import conv

    border = width//2
    filters = _lcn_filters(fmaps, pool_depth, width, sigma) 
    filter_shape = filters.shape
    blurred_mean = conv.conv2d(input=image, filters=filters, 
            image_shape=im_shape, filter_shape=filter_shape,
            border_mode='full')
    image -= blurred_mean[:, :, border:-border, border:-border]
    
    image_sqr = T.sqr(image)
    blurred_sqr = conv.conv2d(input=image_sqr, filters=filters, 
            image_shape=im_shape, filter_shape=filter_shape,
            border_mode='full')

    div = T.sqrt(blurred_sqr[:, :, border:-border, border:-border])
    fm_mean = div.mean(axis=[2, 3])
    div = T.largest(fm_mean.dimshuffle(0, 1, 'x', 'x'), div) + 1e-6
    image = image/div
    return T.cast(image, theano.config.floatX)


def _build_lcns(shape, depth, width, sigma):
    """
    Build _depth_ many local contrast normalizer
    for usage in a pyramid.
    """
    import theano
    import theano.tensor as T
    print "[HELPERS] lcn with depth {0}, width {1}, sigma {2}".format(depth, width, sigma)
    lcns = []
    for i in xrange(depth):
        x = T.matrix("x{0}".format(i))
        x = x.reshape((1, 1, shape[0], shape[1]))
        lcned = _lcn(x, (1, 1, shape[0], shape[1]), fmaps=1, pool_depth=1, width=width, sigma=sigma)
        lcns.append(theano.function([x], lcned, allow_input_downcast=True))
        shape = (shape[0]//2, shape[1]//2)
    print "[HELPERS] lcn done."
    return lcns


def _pyramid(store, key, new, pars):
    """Pyramid store.

    """
    # this needs the cv module from osdf.
    chunk, schema, params = pars[0], pars[1], pars[2]

    depth = 0
    if schema is "Laplace":
        from osdfcv.pyramid.laplacian import build_pil
        build_pyr = build_pil
        depth = params[0]
    elif schema is "LCN":
        from osdfcv.pyramid.lcn import build_pil
        depth, width, sigma = params
        shape = store[key].shape
        dx = int(np.sqrt(shape[1]))
        shape = (dx, dx)
        lcns = _build_lcns(shape, depth, width, sigma)
        from functools import partial
        build_pyr = partial(build_pil, lcns=lcns)
    elif schema is "Fovea":
        from osdfcv.pyramid.fovea import build
        depth = params[0]
        build_pyr = build
    else:
        assert False, "Don't know pyramid schmema %s"%schema

    assert depth > 0, "Need a decent depth: %d"%depth

    # collect inputs in group
    grp = new.create_group(name=key)
    dsets = []
    dtype = store[key].dtype
    shape = store[key].shape
    dx = int(np.sqrt(shape[1]))
    for d in xrange(depth):
        dsets.append(grp.create_dataset(name=key+str(d), shape=shape, dtype=dtype))
        shape = (shape[0], shape[1]/4)

    k = 0 # global counter, not nice
    for i in xrange(0, store[key].shape[0], chunk):
        for l in store[key][i:i+chunk]:
            pyramid = build_pyr(l.reshape(dx, dx), depth)
            for j, img in enumerate(pyramid):
                dsets[j][k] = img.ravel()
            k = k + 1

    for attrs in store.attrs:
        grp.attrs[attrs] = store.attrs[attrs]
    for d in xrange(depth):
        dsets[d].attrs["patch_shape"] = (dx, dx)
        dx = dx/2
    grp.attrs['depth'] = depth
    grp.attrs['schema'] = depth


def _pyramid_fuse(store, key, new, pars):
    """Pyramid store. All stages of a pyramid
    fused into one row vector.

    """
    # this needs the cv module from osdf.
    chunk, schema, shapes = pars[0], pars[1], pars[2]
    depth = len(shapes)

    if schema is "Laplace":
        from osdfcv.pyramid.laplacian import build_pil

    dtype = store[key].dtype
    n, x = store[key].shape
    dx = int(np.sqrt(x))
    shape = 0
    for sh in shapes:
        shape = shape + sh[0]*sh[1]
    dset = new.create_dataset(name=key+"_pyr", shape=(n, shape), dtype=dtype)

    k = 0 # global counter, not nice
    for i in xrange(0, store[key].shape[0], chunk):
        for l in store[key][i:i+chunk]:
            pyramid = build_pil(l.reshape(dx, dx), depth)
            start = 0
            for j, img in enumerate(pyramid):
                stop = start + shapes[j][0]*shapes[j][1]
                dset[k][start:stop] = img.ravel()
                print dset[k]
                start = stop
            k = k + 1
            print k

    for attrs in store.attrs:
        new.attrs[attrs] = store.attrs[attrs]
    new.attrs['depth'] = depth
    new.attrs['schema'] = schema


def _concat(store, key, new, pars):
    """
    """
    chunk = pars[0]

    n = 0
    _tmp = []
    dsets = []
    d = 0
    for ds in store[key]:
        dsets.append(store[key][ds])
        d = d + store[key][ds].shape[1]

    n = dsets[0].shape[0]
    for ds in dsets[1:]:
        assert n == ds.shape[0], "Concatenate needs same number samples for all datasets."

    dset = new.create_dataset(name=key, shape=(n, d), dtype=dsets[0].dtype)

    for i in xrange(0, n, chunk):
        j = 0
        for ds in dsets:
            jj = ds.shape[1]
            dset[i:i+chunk, j:jj] = ds[i:i+chunk]
            j = jj


def _divisive(store, key, new, pars):
    """Divide by row-norm, possibly scale.

    _store_ has to behave like an np.array. Works __inplace__.
    """
    chunk, eps, C = pars[0], pars[1], pars[2]
    
    dset = new.create_dataset(name=key, shape=store[key].shape, dtype=store[key].dtype)

    for i in xrange(0, store[key].shape[0], chunk):
        norm = np.sqrt(np.sum(dset[i:i+chunk]**2, axis=1) + eps)
        dset[i:i+chunk] /= np.atleast_2d(norm).T
        dset[i:i+chunk] *= C

    for attrs in store[key].attrs:
        dset.attrs[attrs] = store[key].attrs[attrs]
    dset.attrs['DivisiveC'] = C

    for attrs in store.attrs:
        new.attrs[attrs] = store.attrs[attrs]
    new.attrs['DivisiveC'] = C


def _row0(store, key, new, chunk=512):
    """Subtract row-mean 
    """
    dset = new.create_dataset(name=key, shape=store[key].shape, dtype=store[key].dtype)

    for i in xrange(0, store[key].shape[0], chunk):
        means = np.mean(store[key][i:i+chunk], axis=1)
        dset[i:i+chunk] = store[key][i:i+chunk] - np.atleast_2d(means).T

    for attrs in store[key].attrs:
        dset.attrs[attrs] = store[key].attrs[attrs]

    for attrs in store.attrs:
        new.attrs[attrs] = store.attrs[attrs]


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

    for attrs in store.attrs:
        new.attrs[attrs] = store.attrs[attrs]
    new.attrs["patch_shape"] = shape


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
    dset.attrs["patch_shape"] = (dy, dx)

    for attrs in store.attrs:
        new.attrs[attrs] = store.attrs[attrs]
    new.attrs["patch_shape"] = (dy, dx)


def _floatify(store, key, float_store, pars):
    """Just dump store into a new store
    that is (i) of dtype float and
    (ii) writeable.
    """
    chunk = pars
    shape = store[key].shape
    dset = float_store.create_dataset(name=key, shape=shape, dtype=np.float)

    for i in xrange(0, shape[0], chunk):
        dset[i:i+chunk] = np.asarray(store[key][i:i+chunk], dtype=np.float)

    for attrs in store[key].attrs:
        dset.attrs[attrs] = store[key].attrs[attrs]


def _double(store, key, new, pars):
    """New store is simply every row twice next to each other
    """
    chunk = pars[0]

    shape = store[key].shape
    dset = new.create_dataset(name=key, shape=(shape[0], 2*shape[1]), dtype=store[key].dtype)

    for i in xrange(0, store[key].shape[0], chunk):
        dset[i:i+chunk, :shape[1]] = store[key][i:i+chunk]
        dset[i:i+chunk, shape[1]:] = store[key][i:i+chunk]

    for attrs in store[key].attrs:
        dset.attrs[attrs] = store[key].attrs[attrs]

    for attrs in store.attrs:
        new.attrs[attrs] = store.attrs[attrs]
