from .gpu import CP


class BaseRegularizer:
    def __init__(self, reg_strength):
        self.reg_strength = float(reg_strength)

    def __call__(self, weights):
        raise NotImplementedError()

    def __repr__(self):
        return f'{type(self).__name__}({self.reg_strength})'


class L1(BaseRegularizer):
    def __call__(self, weights):
        loss = self.reg_strength * CP.cp.sum(CP.cp.abs(weights))
        grad = self.reg_strength * CP.cp.sign(weights)
        return float(loss), grad


class L2(BaseRegularizer):
    def __call__(self, weights):
        loss = self.reg_strength * CP.cp.sum(weights ** 2)
        grad = self.reg_strength * 2 * weights
        return float(loss), grad
