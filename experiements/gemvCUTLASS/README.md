Run this to ge the python extension of the cuda files in this folder


```
CUDA_HOME=/usr/local/cuda-12.8 \
PATH=/usr/local/cuda-12.8/bin:$PATH \
python setup_sparse.py build_ext --inplace
```