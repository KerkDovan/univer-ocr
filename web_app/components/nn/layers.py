import math
from collections import Iterable

import numpy as np

from .initializers import kaiming_uniform
from .optimizers import Adam


def tuplize(name, var, length):
    is_negative = False
    result = None

    if isinstance(var, int):
        is_negative = var < 0
        result = tuple(var for _ in range(length))

    elif isinstance(var, Iterable):
        tmp = tuple(var)
        if len(tmp) == length and all(isinstance(x, int) for x in tmp):
            is_negative = any(x < 0 for x in tmp)
            result = tmp

    if is_negative:
        raise ValueError(f'{name} cannot be negative, found: {var}')
    if result is None:
        raise TypeError(
            f'{name} must be either int or iterable of ints of length {length}, '
            f'found {type(var)}')

    return result


class BaseLayer:
    def __init__(self,
                 name=None,
                 input_shape=None,
                 initializer=kaiming_uniform,
                 regularizer=None,
                 optimizer=Adam()):
        self.name = name
        self.input_shape = input_shape
        self.initializer = initializer
        self.regularizer = regularizer
        self.optimizer = optimizer

        self._mem = {}
        self.single_input = False

    def forward(self, inputs):
        if not isinstance(inputs, list):
            inputs = [inputs]
            self.single_input = True
        result = []
        for mem_id, X in enumerate(inputs):
            result.append(self._forward(X, mem_id))
        if self.single_input:
            return result[0]
        return result

    def backward(self, grads):
        if not isinstance(grads, list):
            grads = [grads]
            self.single_input = True
        result = []
        for mem_id, grad in enumerate(grads):
            result.append(self._backward(grad, mem_id))
        self.clear_memory()
        if self.single_input:
            self.single_input = False
            return result[0]
        return result

    def _forward(self, X, mem_id=0):
        raise NotImplementedError()

    def _backward(self, grad, mem_id=0):
        raise NotImplementedError()

    def update_grads(self):
        for param in self.params().values():
            param.update_grad()

    def clear_grads(self):
        for param in self.params().values():
            param.clear_grad()

    def clear_memory(self):
        self._mem = {}

    def get_output_shape(self, input_shape):
        raise NotImplementedError()

    def params(self):
        return {}

    def count_parameters(self, param=None):
        if param is not None:
            return self.params()[param].value.size
        return sum([param.value.size for param in self.params().values()])

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


class Concat(BaseLayer):
    def __init__(self, axis=-1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.axis = axis

    def forward(self, inputs):
        if not isinstance(inputs, list):
            self.single_input = True
            return inputs
        self._mem = [X.shape for X in inputs]
        result = [np.concatenate(inputs, axis=self.axis)]
        return result

    def backward(self, grads):
        if self.single_input:
            self.single_input = False
            return grads
        shapes = self._mem
        axis = self.axis
        result = []
        prev = [x for x in shapes[0]]
        prev[axis] = 0
        for i, shape in enumerate(shapes):
            slices = [slice(0, c) for c in shape]
            slices[axis] = slice(prev[axis], prev[axis] + shape[axis])
            result.append(grads[0][tuple(slices)])
            prev[axis] += shape[axis]
        return result


class Convolutional2D(BaseLayer):
    def __init__(self, kernel_size, in_channels=None, out_channels=None,
                 padding=0, padding_value=0, stride=1,
                 w=None, bias=True, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.kernel_size = tuplize('kernel_size', kernel_size, 2)
        if in_channels is None:
            self.in_channels = self.input_shape[3]
        else:
            self.in_channels = in_channels
        self.out_channels = out_channels
        self.padding = tuplize('padding', padding, 2)
        self.padding_value = padding_value
        self.stride = tuplize('stride', stride, 2)

        self.w_shape = (np.prod(self.kernel_size) * self.in_channels + 1, self.out_channels)

        self.bias = bias
        if w is None:
            self.w = Param(self.initializer(*self.w_shape), optimizer=self.optimizer)
        else:
            assert w.shape == self.w_shape
            self.w = w
        self._init_optimizer()

    def _forward(self, X, mem_id=0):
        batch_size, height, width, channels = X.shape
        assert channels == self.in_channels

        kh, kw = self.kernel_size
        ph, pw = self.padding
        sh, sw = self.stride

        output_shape = self.get_output_shape(X.shape)
        _, out_height, out_width, _ = output_shape

        if ph != 0 or pw != 0:
            nh, nw = height + 2 * ph, width + 2 * pw
            padded = self.padding_value * np.ones((batch_size, nh, nw, channels))
            padded[:, ph:nh - ph, pw:nw - pw, :] = X
            X = padded

        self._mem[mem_id] = X

        result = np.zeros(output_shape)
        bias_vec = self.bias * np.ones((batch_size, 1))
        new_shape = (batch_size, self.w_shape[0] - 1)

        for y in range(0, out_height, sh):
            for x in range(0, out_width, sw):
                i = np.reshape(X[:, y:y + kh, x:x + kw, :], new_shape)
                i = np.concatenate((i, bias_vec), axis=1)
                res = np.dot(i, self.w.value)
                result[:, y, x, :] = res

        return result

    def _backward(self, grad, mem_id=0):
        X = self._mem[mem_id]
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
        if ph != 0 or pw != 0:
            dx_total = dx_total[:, ph:height - ph, pw:width - pw, :]

        return dx_total

    def get_output_shape(self, input_shape):
        batch_size, height, width, _ = input_shape

        kh, kw = self.kernel_size
        ph, pw = self.padding
        sh, sw = self.stride

        out_height = math.floor((height + 2 * ph - (kh - 1) - 1) / sh + 1)
        out_width = math.floor((width + 2 * pw - (kw - 1) - 1) / sw + 1)

        return batch_size, out_height, out_width, self.out_channels

    def params(self):
        return {'w': self.w}


class Flatten(BaseLayer):
    def _forward(self, X, mem_id=0):
        self._mem[mem_id] = X.shape
        return np.reshape(X, self.get_output_shape(X.shape))

    def _backward(self, grad, mem_id=0):
        reshaped = np.reshape(grad, self._mem[mem_id])
        return reshaped

    def get_output_shape(self, input_shape):
        return input_shape[0], np.product(input_shape[1:])


class FullyConnected(BaseLayer):
    def __init__(self, n_input=None, n_output=None, w=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if n_input is None:
            self.n_input = self.input_shape[1]
        else:
            self.n_input = n_input
        self.n_output = n_output
        if w is None:
            self.w = Param(
                self.initializer(self.n_input + 1, self.n_output),
                optimizer=self.optimizer)
        else:
            assert w.shape == (self.n_input + 1, self.n_output)
            self.w = Param(np.copy(w), optimizer=self.optimizer)
        self._init_optimizer()

    def _forward(self, X, mem_id=0):
        nx = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
        self._mem[mem_id] = nx
        y = np.dot(nx, self.w.value)
        return y

    def _backward(self, grad, mem_id=0):
        wt = np.transpose(self.w.value[:-1, :])
        xt = np.transpose(self._mem[mem_id])
        dx = np.dot(grad, wt)
        dw = np.dot(xt, grad)
        self.w.grad += dw
        return dx

    def get_output_shape(self, input_shape):
        return input_shape[0], self.n_output

    def params(self):
        return {'w': self.w}


class Input(BaseLayer):
    def __init__(self, input_shape=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if isinstance(input_shape, int):
            self.input_shape = (input_shape,)
        elif isinstance(input_shape, tuple):
            self.input_shape = input_shape
        else:
            raise TypeError('input_shape must be int or tuple')

    def _forward(self, X, mem_id=0):
        assert X.shape[1:] == self.input_shape[1:]
        return X

    def _backward(self, grad, mem_id=0):
        return grad

    def get_output_shape(self, input_shape):
        return input_shape


class MaxPool2D(BaseLayer):
    def __init__(self, kernel_size, padding=0, stride=None, ceil_mode=False, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.kernel_size = tuplize('kernel_size', kernel_size, 2)
        self.padding = tuplize('padding', padding, 2)
        self.stride = self.kernel_size if stride is None else tuplize('stride', stride)
        self.ceil_mode = ceil_mode

    def _forward(self, X, mem_id=0):
        batch_size, height, width, channels = X.shape

        kh, kw = self.kernel_size
        ph, pw = self.padding
        sh, sw = self.stride

        output_shape = self.get_output_shape(X.shape)
        _, out_height, out_width, _ = output_shape

        if ph != 0 or pw != 0:
            nh, nw = height + 2 * ph, width + 2 * pw
            padded = np.zeros((batch_size, nh, nw, channels))
            padded[:, ph:nh - ph, pw:nw - pw, :] = X
            X = padded

        result = np.zeros(output_shape)
        mask = np.zeros((batch_size, kh * out_height, kw * out_width, channels))

        for y in range(out_height):
            for x in range(out_width):
                iy, ix = y * sh, x * sw
                i = X[:, iy:iy + kh, ix:ix + kw, :]
                max_el = np.max(i, axis=(1, 2))
                max_el = np.reshape(max_el, (batch_size, 1, 1, channels))
                submask = i == max_el
                _, mh, mw, _ = submask.shape
                my, mx = kh * y, kw * x
                mask[:, my:my + mh, mx:mx + mw, :] = submask
                result[:, y, x, :] += np.reshape(max_el, (batch_size, channels))

        self._mem[mem_id] = mask, X.shape
        return result

    def _backward(self, grad, mem_id=0):
        mask, input_shape = self._mem[mem_id]
        batch_size, height, width, channels = input_shape
        _, out_height, out_width, _ = grad.shape

        kh, kw = self.kernel_size
        ph, pw = self.padding
        sh, sw = self.stride

        result = np.zeros(input_shape)

        for y in range(out_height):
            for x in range(out_width):
                ty, tx = sh * y, sw * x
                my, mx = kh * y, kw * x
                target = result[:, ty:ty + kh, tx:tx + kw, :]
                _, th, tw, _ = target.shape
                sub_shape = (batch_size, 1, 1, channels)
                submask = mask[:, my:my + th, mx:mx + tw, :]
                subgrad = np.reshape(grad[:, y, x, :], sub_shape)
                subsum = np.reshape(np.sum(submask, axis=(1, 2)), sub_shape)
                target += (subgrad / subsum) * submask

        if ph != 0 or pw != 0:
            result = result[:, ph:height - ph, pw:width - pw, :]

        return result

    def get_output_shape(self, input_shape):
        batch_size, height, width, channels = input_shape

        kh, kw = self.kernel_size
        ph, pw = self.padding
        sh, sw = self.stride
        ceil = math.ceil if self.ceil_mode else math.floor

        out_height = ceil((height + 2 * ph - (kh - 1) - 1) / sh + 1)
        out_width = ceil((width + 2 * pw - (kw - 1) - 1) / sw + 1)

        return batch_size, out_height, out_width, channels


class Relu(BaseLayer):
    def _forward(self, X, mem_id=0):
        self._mem[mem_id] = X >= 0
        return X * self._mem[mem_id]

    def _backward(self, grad, mem_id=0):
        result = grad * self._mem[mem_id]
        return result

    def get_output_shape(self, input_shape):
        return input_shape


class Upsample2D(BaseLayer):
    def __init__(self, scale_factor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scale_factor = tuplize('scale_factor', scale_factor, 2)

    def _forward(self, X, mem_id=0):
        self._mem[mem_id] = X.shape
        return X.repeat(self.scale_factor[0], axis=1).repeat(self.scale_factor[1], axis=2)

    def _backward(self, grad, mem_id=0):
        batch_size, height, width, channels = grad.shape
        sf_y, sf_x = self.scale_factor

        result = np.zeros(self._mem[mem_id])
        for y, gy in enumerate(range(0, height, sf_y)):
            for x, gx in enumerate(range(0, width, sf_x)):
                region = grad[:, gy:gy + sf_y, gx:gx + sf_x, :]
                result[:, y, x, :] += np.sum(region, axis=(1, 2))

        return result

    def get_output_shape(self, input_shape):
        return tuple(np.array(input_shape) * (1, *self.scale_factor, 1))
