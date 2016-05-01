The mnist dataset, available from
http://deeplearning.net/tutorial/gettingstarted.html#index-1

If you don't have mnist.h5 in this directory:
- cd to this directory
- download mnist.pkl.gz from [deeplearning.net]:
  ```wget http://deeplearning.net/data/mnist/mnist.pkl.gz```
- ```python dataset.py``` (this builds (default filename) ```mnist.h5```)

mnist.h5 has three groups, "train", "validation" and "test".
Every group has two datasets, "inputs" and "targets".


   [deeplearning.net]: http://deeplearning.net/
