import dataset as ds

print "'python dataset.py' must have been run before."
print "Patience. And enough storage!"
ds.build_supervised_store()
print "A bit more patience."
store = ds.get_store()
ds.build_evaluate_store(store, pair_list=(50000,))
print "Please generate 3 symbolic links:"
print "ln -s evaluate_liberty_64x64.h5 supervised_50000_evaluate_liberty_64x64.h5"
print "ln -s evaluate_notredame_64x64.h5 supervised_50000_evaluate_notredame_64x64.h5"
print "ln -s evaluate_yosemite_64x64.h5 supervised_50000_evaluate_yosemite_64x64.h5"
