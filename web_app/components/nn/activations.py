import numpy as np


class BaseActivationFunction:
    def __init__(self):
        pass

    def __call__(self):
        raise NotImplementedError()


class Relu(BaseActivationFunction):
    def __call__(self, X):
        return X * (X >= 0)


class Sigmoid(BaseActivationFunction):
    def __call__(self, X):
        return np.array([1 / (1 + np.math.exp(-a)) for a in X])


class Softmax(BaseActivationFunction):
    """https://peterroelants.github.io/posts/cross-entropy-softmax/"""
    def __call__(self, X):
        ex = np.exp(X - np.max(X))
        return ex / np.sum(ex)
