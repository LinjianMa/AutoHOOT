This folder includes the implementation of various backends.

NumPy backend:

* Supports operations on dense tensors and sparse tensors (via sparse library).

* Currently only supports the COO sparse format for sparse tensors.

* Einsum operation is executed via opt_einsum, which is compatible with both NumPy and sparse.

CTF, JAX, and TensorFlow backends:

* Currently only support operations on dense tensors.
