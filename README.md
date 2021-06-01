# AutoHOOT: Automatic High-Order Optimization for Tensors

AutoHOOT is a Python-based automatic differentiation framework targeting at high-order optimization for large scale tensor computations.

AutoHOOT contains a new explicit Jacobian / Hessian expression generation kernel whose outputs keep the input tensors’ granularity and are easy to optimize. It also contains a new computational graph optimizer that combines both the traditional optimization techniques for compilers and techniques based on specific tensor algebra. The optimization module generates expressions as good as manually written codes in other frameworks for the numerical algorithms of tensor computations.

The library is compatible with other AD libraries, including [TensorFlow](https://github.com/tensorflow/tensorflow) and [Jax](https://github.com/google/jax), and numerical libraries, including [NumPy](https://github.com/numpy/numpy) and [Cyclops Tensor Framework](https://github.com/cyclops-community/ctf).

The example usage of the libaray is shown in the [examples](https://github.com/LinjianMa/AutoHOOT/tree/master/examples) and [tests](https://github.com/LinjianMa/AutoHOOT/tree/master/tests) folders.


## Installation
Considering this package is in development, it is recommended to install it in
the editable mode.

```bash
    pip install -e path/to/the/project/directory
```


## Tests cases
Run all tests with
```bash
# sudo pip install pytest
python -m pytest tests/*.py
# You can specify the python file as well.
```

```bash
# Run with specific backends
pytest tests/autodiff_test.py --backendopt numpy jax
```


## Notations
This repo involves a lot of tensor contraction operations. We use the following notations in our documentation:

For the matrix multiplication `C = A @ B`, the following expressions are equal:

```
C = einsum("ij,jk->ik", A, B)
```
```
C["ik"] = A["ij"] * B["jk"]
```
```
C[0,2] = A[0,1] * B[1,2]
```

## Overview of Module API and Data Structures

Suppose our expression is y = x1 @ x2 + x1, we first define our variables x1 and x2 symbolically,
```python
import autodiff as ad

x1 = ad.Variable(name = "x1", shape=[3, 3])
x2 = ad.Variable(name = "x2", shape=[3, 3])
```
Then, you can define the symoblic expression for y,
```python
y = ad.einsum("ab,bc->ac", x1, x2) + x1
```
With this computation graph, we can evaluate the value of y given any values of x1 and x2: simply walk the graph in a topological order, and for each node, use its associated operator to compute an output value given input values. The evaluation is done in Executor.run method.
```python
executor = ad.Executor([y])
y_val = executor.run(feed_dict = {x1 : x1_val, x2 : x2_val})
```
If we want to evaluate the gradients of y with respect to x1 and x2, as we would often do for loss function wrt parameters in usual machine learning training steps, we need to construct the gradient nodes, grad_x1 and grad_x2.
```python
grad_x1, grad_x2 = ad.gradients(y, [x1, x2])
```
Once we construct the gradients node, and have references to them, we can evaluate the gradients using Executor as before,
```python
executor = ad.Executor([y, grad_x1, grad_x2])
y_val, grad_x1_val, grad_x2_val = executor.run(feed_dict = {x1 : x1_val, x2 : x2_val})
```
grad_x1_val, grad_x2_val now contain the values of dy/dx1 and dy/dx2.

## Second-order information

This repo also supports Hessian-vector products through reverse-mode autodiff. As to a expression `y = x^T @ H @ x`, we first define the expression 
```python
x = ad.Variable(name="x", shape=[3, 1])
H = ad.Variable(name="H", shape=[3, 3])
v = ad.Variable(name="v", shape=[3, 1])
y = ad.sum(ad.einsum("ab,bc->ac", ad.einsum("ab,bc->ac", ad.transpose(x), H), x))
```
Then define the expression for the gradient and Hessian-vector product:
```python
grad_x, = ad.gradients(y, [x])
Hv, = ad.hvp(output_node=y, node_list=[x], vector_list=[v])
```
Then we can evaluate y, grad_x and Hv all at once:
```python
y_val, grad_x_val, Hv_val = executor.run(feed_dict={
    x: x_val, H: H_val, v: v_val
    })
```

## Source code generation

This repo also supports source code generation for the constructed computational graphs. See [the test file](https://github.com/LinjianMa/AutoHOOT/blob/master/tests/s2s_test.py) for example usage.


## Acknowledging Usage

The library is available to everyone. If you would like to acknowledge the usage of the library, please cite the following paper:

Linjian Ma, Jiayu Ye, and Edgar Solomonik. AutoHOOT: Automatic High-Order Optimization for Tensors. International Conference on Parallel Architectures and Compilation Techniques (PACT), October 2020.
