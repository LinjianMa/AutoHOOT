# Reverse-mode Automatic Differentiation

## Tests cases
Run all tests with
```bash
# sudo pip install nose
nosetests -v autodiff_test.py
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
