import numpy as np


class Regularizer:
    def __init__(self, reg_strength):
        self.reg_strength = float(reg_strength)

    def __call__(self, weights):
        raise NotImplementedError()

    def __repr__(self):
        return f'{type(self).__name__}({self.reg_strength})'


class L1(Regularizer):
    def __call__(self, weights):
        loss = self.reg_strength * np.sum(np.abs(weights))
        grad = self.reg_strength * np.sign(weights)
        return loss, grad


class L2(Regularizer):
    def __call__(self, weights):
        loss = self.reg_strength * np.sum(weights ** 2)
        grad = self.reg_strength * 2 * weights
        return loss, grad
