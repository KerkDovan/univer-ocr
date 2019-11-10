import math

import numpy as np

from .initializers import kaiming_uniform
from .optimizers import Adam


class Layer:
    def __init__(self,
                 initializer=kaiming_uniform,
                 regularizer=None,
                 optimizer=Adam()):
        self.initializer = initializer
        self.regularizer = regularizer
        self.optimizer = optimizer

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
            self.optimizer.add_param(param)


class Param:
    def __init__(self, value, optimizer=None):
        self.value = value
        self.grad = np.zeros_like(value)
        self.optimizer = optimizer
        self.optimizer.add_param(self)

    def update_grad(self):
        self.optimizer.update(self)

    def clear_grad(self):
        self.grad = np.zeros_like(self.value)


class Convolutional2D(Layer):
    def __init__(self, kernel_size, in_channels, out_channels,
                 padding=0, padding_value=0, stride=1,
                 w=None, bias=True, *args, **kwargs):
        super().__init__(*args, **kwargs)

        def tuplize(name, var):
            if isinstance(var, int):
                return (var, var)
            if isinstance(var, tuple) and len(var) == 2:
                return var
            raise ValueError(f'{name} must be either int or tuple of length 2')

        self.kernel_size = tuplize('kernel_size', kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.padding = tuplize('padding', padding)
        self.padding_value = padding_value
        self.stride = tuplize('stride', stride)

        self.w_shape = (np.prod(self.kernel_size) * self.in_channels + 1, self.out_channels)

        self.bias = bias
        if w is None:
            self.w = Param(self.initializer(*self.w_shape), optimizer=self.optimizer)
        else:
            assert w.shape == self.w_shape
            self.w = w
        self._init_optimizer()

    def forward(self, X):
        batch_size, height, width, channels = X.shape
        assert channels == self.in_channels

        kh, kw = self.kernel_size
        ph, pw = self.padding
        sh, sw = self.stride

        out_height = math.floor((height + 2 * ph - (kh - 1) - 1) / sh + 1)
        out_width = math.floor((width + 2 * pw - (kw - 1) - 1) / sw + 1)

        if ph != 0 or pw != 0:
            nh, nw = height + 2 * ph, width + 2 * pw
            padded = self.padding_value * np.ones((batch_size, nh, nw, channels))
            padded[:, ph:nh - ph, pw:nw - pw, :] = X
            X = padded

        self._mem = X

        result = np.zeros((batch_size, out_height, out_width, self.out_channels))
        bias_vec = self.bias * np.ones((batch_size, 1))
        new_shape = (batch_size, self.w_shape[0] - 1)

        for y in range(0, out_height, sh):
            for x in range(0, out_width, sw):
                i = np.reshape(X[:, y:y + kh, x:x + kw, :], new_shape)
                i = np.concatenate((i, bias_vec), axis=1)
                res = np.dot(i, self.w.value)
                result[:, y, x, :] = res

        return result

    def backward(self, grad):
        X = self._mem
        batch_size, height, width, in_channels = X.shape
        _, out_height, out_width, out_channels = grad.shape

        kh, kw = self.kernel_size
        ph, pw = self.padding
        sh, sw = self.stride

        bias_vec = self.bias * np.ones((batch_size, 1))
        new_shape = (batch_size, self.w_shape[0] - 1)
        dw_total = np.zeros_like(self.w.value)
        dx_total = np.zeros_like(X)
        wt = np.transpose(self.w.value[:-1, :])

        for y in range(0, out_height, sh):
            for x in range(0, out_width, sw):
                cur_grad = grad[:, y, x, :]

                xt = np.reshape(X[:, y:y + kh, x:x + kw, :], new_shape)
                xt = np.concatenate((xt, bias_vec), axis=1)
                xt = np.transpose(xt)
                dw = np.dot(xt, cur_grad)
                dw_total += dw

                dx = np.dot(cur_grad, wt)
                dx = np.reshape(dx, (batch_size, kh, kw, in_channels))
                dx_total[:, y:y + kh, x:x + kw, :] += dx

        self.w.grad += dw_total
        dx_total = dx_total[:, ph:height - ph, pw:width - pw, :]

        self.clear_memory()
        return dx_total

    def params(self):
        return {'w': self.w}


class Flatten(Layer):
    def forward(self, X):
        self._mem = X.shape
        return np.reshape(X, (X.shape[0], np.product(X.shape[1:])))

    def backward(self, grad):
        reshaped = np.reshape(grad, self._mem)
        self.clear_memory()
        return reshaped


class FullyConnected(Layer):
    def __init__(self, n_input, n_output, w=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_input = n_input
        self.n_output = n_output
        if w is None:
            self.w = Param(self.initializer(n_input + 1, n_output), optimizer=self.optimizer)
        else:
            assert w.shape == (n_input + 1, n_output)
            self.w = Param(np.copy(w), optimizer=self.optimizer)
        self._init_optimizer()

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
