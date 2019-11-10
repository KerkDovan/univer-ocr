import numpy as np


class ActivationFunction:
    def __init__(self):
        pass

    def __call__(self):
        raise NotImplementedError()


class Relu(ActivationFunction):
    def __call__(self, X):
        return X * (X >= 0)


class Sigmoid(ActivationFunction):
    def __call__(self, X):
        return np.array([1 / (1 + np.math.exp(-a)) for a in X])


class Softmax(ActivationFunction):
    """https://peterroelants.github.io/posts/cross-entropy-softmax/"""
    def __call__(self, X):
        ex = np.exp(X - np.max(X))
        return ex / np.sum(ex)
