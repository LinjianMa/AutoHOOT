# Reverse-mode Automatic Differentiation

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

Suppose our expression is y=x1*x2+x1, we first define our variables x1 and x2 symbolically,
```python
import autodiff as ad

x1 = ad.Variable(name = "x1")
x2 = ad.Variable(name = "x2")
```
Then, you can define the symoblic expression for y,
```python
y = x1 * x2 + x1
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
x = ad.Variable(name="x")
H = ad.Variable(name="H")
v = ad.Variable(name="v")
y = ad.transpose(x) @ H @ x
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

This repo also supports source code generation for both gradients and Hessian-vector products. For the same expression above, we can generate the functions for gradient and Hvp calculations as follows:
```python
from source import SourceToSource

StS = SourceToSource()
StS.forward(y, file=open("example_forward.py", "w"))
StS.gradients(y, [x], file=open("example_grad.py", "w"))
StS.hvp(y, [x], [v], file=open("example_hvp.py", "w"))
```
We can then use the generated functions as follows:
```python
import example_forward, example_grad, example_hvp

y_val_s2s = example_forward.forward([x_val, H_val])
grad_x_val_s2s, = example_grad.gradients([x_val, H_val])
Hv_val_s2s, = example_hvp.hvp([x_val, H_val, v_val])
```
