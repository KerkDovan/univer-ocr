from collections import namedtuple

from .gpu import CP


class State:
    def __init__(self, attributes, defaults):
        for attr, default in zip(attributes, defaults):
            setattr(self, attr, default)


class BaseOptimizer:
    def __init__(self):
        self.groups = {}

    def add_param(self, param):
        self.groups[id(param)] = namedtuple('Group', 'param state')(param, self._init_state())

    def update(self, param):
        raise NotImplementedError()

    def _get_group(self, param):
        return self.groups[id(param)]

    def _init_state(self):
        raise NotImplementedError()


class Adagrad(BaseOptimizer):
    def __init__(self, lr=0.01, initial_accumulated=0):
        super().__init__()
        self.lr = lr
        self.initials = [initial_accumulated]

    def update(self, param):
        state = self._get_group(param).state
        state.accumulated += param.grad ** 2
        adaptive_lr = state.lr / CP.cp.sqrt(state.accumulated)
        param.value -= adaptive_lr * param.grad

    def _init_state(self):
        return State(['accumulated'], self.initials)


class Adam(BaseOptimizer):
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999,
                 initial_velocity=0, initial_accumulated=0):
        super().__init__()
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.initials = [initial_velocity, initial_accumulated]

    def update(self, param):
        state = self._get_group(param).state
        state.velocity = self.beta1 * state.velocity + (1 - self.beta1) * param.grad
        state.accumulated = self.beta2 * state.accumulated + (1 - self.beta2) * param.grad ** 2
        adaptive_lr = self.lr / CP.cp.sqrt(state.accumulated)
        param.value -= adaptive_lr * state.velocity

    def _init_state(self):
        return State(['velocity', 'accumulated'], self.initials)


class Momentum(BaseOptimizer):
    def __init__(self, lr, momentum=0, initial_velocity=0):
        super().__init__()
        self.lr = lr
        self.momentum = momentum
        self.velocity = 0
        self.initials = [initial_velocity]

    def update(self, param):
        state = self._get_group(param).state
        state.velocity = self.momentum * state.velocity - self.lr * param.grad
        param.value += state.velocity

    def _init_state(self):
        return State(['velocity'], self.initials)


class RMSProp(BaseOptimizer):
    def __init__(self, lr=0.01, rho=0.99, initial_accumulated=0):
        super().__init__()
        self.lr = lr
        self.rho = rho
        self.initials = [initial_accumulated]

    def update(self, param):
        state = self._get_group(param).state
        state.accumulated = self.rho * state.accumulated + (1 - self.rho) * param.grad ** 2
        adaptive_lr = self.lr / CP.cp.sqrt(state.accumulated)
        param.value -= adaptive_lr * param.grad

    def _init_state(self):
        return State(['accumulated'], self.initials)
