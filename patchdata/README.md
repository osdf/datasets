These are the patches downloadable from
[http://www.cs.ubc.ca/~mbrown/patchdata/patchdata.html].

The three datasets ```liberty```, ```notredame``` and ```yosemite```:
- ```wget -c http://matthewalunbrown.com/patchdata/liberty.zip```
- ```wget -c http://matthewalunbrown.com/patchdata/notredame.zip```
- ```wget -c http://matthewalunbrown.com/patchdata/yosemite.zip```


Extract all three zip files into ```liberty/```, ```notredame/```,
```yosemite/``` respectively.

Run ```python dataset.py``` to build a hdf5 store (default name:
```patchdata_64x64.h5```), the size is about 6GB (saved as uint8!).
Use ```select``` to build a store useful for training. Look into
```evaluate.py``` in order to evaluate pairs on ground truth.
