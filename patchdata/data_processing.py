import dataset as ds

#print "'python dataset.py' must have been run before."
#print "Patience. And enough storage!"
#ds.build_supervised_store()
#
#print "A bit more patience."
#store = ds.get_store()
#ds.build_evaluate_store(store, pair_list=(50000,))
#
#print "Please generate 3 symbolic links:"
#print "ln -s evaluate_liberty_64x64.h5 supervised_50000_evaluate_liberty_64x64.h5"
#print "ln -s evaluate_notredame_64x64.h5 supervised_50000_evaluate_notredame_64x64.h5"
#print "ln -s evaluate_yosemite_64x64.h5 supervised_50000_evaluate_yosemite_64x64.h5"


# Prepare 32x32 datasets, with focus on liberty (supervised) training set.
# Next call builds all evaluates with 32x32, liberty supervised fused with 32x32
# and selects from liberty set 400000 samples and resizes them to 32x32.
# Additional processing, e.g. normalization must take place separately.
#ds.build_resize_stores(shape=32, dset=['liberty'], 
#        evaluate=True, supervised=True, sz=250000, 
#        selecting=True, samples=400000)


# Stationarize selected stores
# first, all 32x32 evaluate stores
for d in ["liberty", "notredame", "yosemite"]:
    store = ds.get_store("supervised_50000_evaluate_{0}_32x32.h5".format(d))
    stat = ds.stationary_store(store, C=1, cache=True, exclude=["targets"])
    store.close()
    stat.close()
#
store = ds.get_store("liberty_400000_32x32.select.h5")
stat = ds.stationary_store(store, C=1, cache=True, exclude=["targets"])
store.close()
stat.close()
#
store = ds.get_store("supervised_250000_evaluate_liberty_fuse_32x32.h5")
stat = ds.stationary_store(store, C=1, cache=True, exclude=["targets"])
store.close()
stat.close()
print "Newly generated *stat.h5 files must be symlinked into working CNN directory!"
