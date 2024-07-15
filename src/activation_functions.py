import numpy as np

def linear(x):
    return x

def linear_derivative(x):
    return np.ones_like(x)

def relu(x):
    return np.maximum(0, x)

# subgradient of relu
def relu_derivative(x):
    return np.where(x >= 0, 1, 0)

def derivative(function):
    if function==linear:
        return linear_derivative
    if function==relu:
        return relu_derivative