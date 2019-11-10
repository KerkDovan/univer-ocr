import numpy as np

from nn.initializers import kaiming_uniform
from nn.optimizers import Momentum


class Layer:
    def __init__(self, reg=None, opt=None):
        self.reg = reg
        self.opt = opt
        self._mem = None

    def forward(self, X):
        raise NotImplementedError()

    def backward(self, grad):
        raise NotImplementedError()

    def update_grads(self):
        for param in self.params().values():
            param.update_grad()

    def clear_grads(self):
        for param in self.params().values():
            param.clear_grad()

    def clear_memory(self):
        self._mem = None

    def params(self):
        return {}

    def regularize(self, weights):
        if self.reg is None:
            return 0
        total_loss = 0
        for param in self.params().values():
            loss, grad = self.reg(param.value)
            param.grad += grad
            total_loss += loss
        return total_loss

    def _init_optimizer(self):
        for param in self.params():
            self.opt.add_param(param)


class Param:
    def __init__(self, value, optimizer=Momentum):
        self.value = value
        self.grad = np.zeros_like(value)
        self.optimizer = optimizer
        self.optimizer.add_param(self)

    def update_grad(self):
        self.optimizer.update(self)

    def clear_grad(self):
        self.grad = np.zeros_like(self.value)


class Convolutional2D(Layer):
    def __init__(self, filters, sizes, padding=None, strides=(1, 1)):
        super().__init__()
        self.sizes = sizes
        self.strides = strides

    def forward(self, X):
        pass

    def backward(self, grad):
        pass


class Flatten(Layer):
    def forward(self, X):
        self._mem = X.shape
        return np.reshape(X, (X.shape[0], np.product(X.shape[1:])))

    def backward(self, grad):
        reshaped = np.reshape(grad, self._mem)
        self.clear_memory()
        return reshaped


class FullyConnected(Layer):
    def __init__(self, n_input, n_output, w=None, b=None, reg=None):
        super().__init__(reg=reg)
        self.n_input = n_input
        self.n_output = n_output
        if w is None:
            self.w = Param(kaiming_uniform(n_input + 1, n_output))
        else:
            assert w.shape == (n_input + 1, n_output)
            self.w = Param(np.copy(w))

    def forward(self, X):
        nx = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
        self._mem = nx
        y = np.dot(nx, self.w.value)
        return y

    def backward(self, grad):
        wt = np.transpose(self.w.value[:-1, :])
        xt = np.transpose(self._mem)
        dx = np.dot(grad, wt)
        dw = np.dot(xt, grad)
        self.w.grad += dw
        self.clear_memory()
        return dx

    def params(self):
        return {'w': self.w}


class Relu(Layer):
    def forward(self, X):
        self._mem = X >= 0
        return X * self._mem

    def backward(self, grad):
        result = grad * self._mem
        self.clear_memory()
        return result
