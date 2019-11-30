import math

import numpy as np

from ..gpu import CP
from ..help_func import make_list_if_not, tuplize
from ..initializers import kaiming_uniform
from ..optimizers import Adam
from ..progress_tracker import BaseProgressTracker, track_this


class Param:
    def __init__(self, value, optimizer=None):
        self.value = CP.cp.array(value)
        self.grad = CP.cp.zeros_like(self.value)
        self.optimizer = optimizer
        self.optimizer.add_param(self)

    def update_grad(self):
        self.optimizer.update(self)

    def clear_grad(self):
        self.grad = CP.cp.zeros_like(self.value)


class BaseLayer:
    def __init__(self,
                 name=None,
                 input_shapes=None,
                 trainable=True,
                 initializer=kaiming_uniform,
                 regularizer=None,
                 optimizer=Adam()):
        self.name = name
        self.input_shapes = input_shapes
        self.trainable = trainable
        self.initializer = initializer
        self.regularizer = regularizer
        self.optimizer = optimizer

        self.is_initialized = True

        self._mem = {}

        self.progress_tracker = BaseProgressTracker()

    def initialize_from_X(self, X):
        X = make_list_if_not(X)
        self.initialize([x.shape for x in X])

    def initialize(self, input_shapes):
        self.input_shapes = input_shapes
        self.is_initialized = True

    @track_this('forward')
    def forward(self, inputs):
        assert self.is_initialized, 'You must initialize() layer before calling forward() method'
        inputs = make_list_if_not(inputs)
        result = []
        for mem_id, X in enumerate(inputs):
            result.append(self._forward(X, mem_id))
        return result

    @track_this('backward')
    def backward(self, grads):
        grads = make_list_if_not(grads)
        result = []
        for mem_id, grad in enumerate(grads):
            result.append(self._backward(grad, mem_id))
        self.clear_memory()
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

    def get_all_output_shapes(self, input_shapes):
        return self.get_output_shapes(input_shapes), {}

    def get_output_shapes(self, input_shapes):
        input_shapes = make_list_if_not(input_shapes)
        raise NotImplementedError()

    def params(self):
        return {}

    def get_weights(self):
        return {name: param.value.tolist() for name, param in self.params().items()}

    def set_weights(self, weights):
        for name, param in self.params().items():
            cur_weights = CP.cp.array(np.array(weights.get(name, None)))
            if CP.cp.any(CP.cp.isnan(cur_weights)):
                cur_weights = None
                print(f'{self.name}/{name}: NaN found in loaded weights, skipping')
            if cur_weights is None:
                continue
            param.value = CP.cp.array(cur_weights)

    def count_parameters(self, param=None):
        if param is not None:
            return self.params()[param].value.size
        return sum([param.value.size for param in self.params().values()])

    def regularize(self):
        if self.regularizer is None:
            return 0
        total_loss = 0
        for param in self.params().values():
            loss, grad = self.regularizer(param.value)
            param.grad += grad
            total_loss += loss
        return total_loss

    def _set_name(self, name):
        self.name = name

    def _init_optimizer(self):
        for param in self.params():
            self.optimizer.add_param(param)

    def init_progress_tracker(self, progress_tracker, set_names_recursively=False):
        self.progress_tracker = progress_tracker
        self.progress_tracker.register_layer(self.name)


class Concat(BaseLayer):
    def __init__(self, axis=-1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.axis = axis

    @track_this('forward')
    def forward(self, inputs):
        if not isinstance(inputs, list):
            self._mem = [inputs.shape]
            return inputs
        self._mem = [X.shape for X in inputs]
        result = [CP.cp.concatenate(inputs, axis=self.axis)]
        return result

    @track_this('backward')
    def backward(self, grads):
        grads = make_list_if_not(grads)
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
        self.clear_memory()
        return result

    def get_output_shapes(self, input_shapes):
        input_shapes = np.array(make_list_if_not(input_shapes))
        result = [x for x in input_shapes[0]]
        tmp = np.sum(input_shapes[:, 1:], axis=0)
        result[self.axis] = [input_shapes[0][0], *tmp][self.axis]
        return [tuple(result)]


class Flatten(BaseLayer):
    def _forward(self, X, mem_id=0):
        self._mem[mem_id] = X.shape
        return CP.cp.reshape(X, self.get_output_shapes(X.shape)[0])

    def _backward(self, grad, mem_id=0):
        reshaped = CP.cp.reshape(grad, self._mem[mem_id])
        return reshaped

    def get_output_shapes(self, input_shapes):
        input_shapes = make_list_if_not(input_shapes)
        return [(input_shapes[0][0], np.product(input_shapes[0][1:]))]


class FullyConnected(BaseLayer):
    def __init__(self, n_input=None, n_output=None, w=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_input = n_input
        self.n_output = n_output
        self.w = w
        if self.input_shapes is None and n_input is not None:
            self.input_shapes = [(None, self.n_input)]
        if self.input_shapes is not None:
            self.initialize(self.input_shapes)
        else:
            self.is_initialized = False

    def initialize(self, input_shapes):
        self.input_shapes = input_shapes
        self.n_input = self.input_shapes[0][1]
        if self.n_output is None:
            self.n_output = self.n_input
        if self.w is None:
            self.w = Param(
                self.initializer(self.n_input + 1, self.n_output),
                optimizer=self.optimizer)
        else:
            assert self.w.shape == (self.n_input + 1, self.n_output)
            self.w = Param(CP.cp.copy(self.w), optimizer=self.optimizer)
        self._init_optimizer()
        self.is_initialized = True

    def _forward(self, X, mem_id=0):
        nx = CP.cp.concatenate((X, CP.cp.ones((X.shape[0], 1))), axis=1)
        self._mem[mem_id] = nx
        y = CP.cp.dot(nx, self.w.value)
        return y

    def _backward(self, grad, mem_id=0):
        wt = CP.cp.transpose(self.w.value[:-1, :])
        xt = CP.cp.transpose(self._mem[mem_id])
        dx = CP.cp.dot(grad, wt)
        dw = CP.cp.dot(xt, grad)
        self.w.grad += dw
        return dx

    def get_output_shapes(self, input_shapes):
        input_shapes = make_list_if_not(input_shapes)
        return [(input_shapes[0][0], self.n_output)]

    def params(self):
        return {'w': self.w}


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

        output_shape = self.get_output_shapes(X.shape)[0]
        _, out_height, out_width, _ = output_shape

        if ph != 0 or pw != 0:
            nh, nw = height + 2 * ph, width + 2 * pw
            padded = CP.cp.zeros((batch_size, nh, nw, channels))
            padded[:, ph:nh - ph, pw:nw - pw, :] = X
            X = padded

        result = CP.cp.zeros(output_shape)
        mask = CP.cp.zeros((batch_size, kh * out_height, kw * out_width, channels))

        for y in range(out_height):
            for x in range(out_width):
                iy, ix = y * sh, x * sw
                i = X[:, iy:iy + kh, ix:ix + kw, :]
                max_el = CP.cp.max(i, axis=(1, 2))
                max_el = CP.cp.reshape(max_el, (batch_size, 1, 1, channels))
                submask = i == max_el
                _, mh, mw, _ = submask.shape
                my, mx = kh * y, kw * x
                mask[:, my:my + mh, mx:mx + mw, :] = submask
                result[:, y, x, :] += CP.cp.reshape(max_el, (batch_size, channels))

        self._mem[mem_id] = mask, X.shape
        return result

    def _backward(self, grad, mem_id=0):
        mask, input_shape = self._mem[mem_id]
        batch_size, height, width, channels = input_shape
        _, out_height, out_width, _ = grad.shape

        kh, kw = self.kernel_size
        ph, pw = self.padding
        sh, sw = self.stride

        result = CP.cp.zeros(input_shape)

        for y in range(out_height):
            for x in range(out_width):
                ty, tx = sh * y, sw * x
                my, mx = kh * y, kw * x
                target = result[:, ty:ty + kh, tx:tx + kw, :]
                _, th, tw, _ = target.shape
                sub_shape = (batch_size, 1, 1, channels)
                submask = mask[:, my:my + th, mx:mx + tw, :]
                subgrad = CP.cp.reshape(grad[:, y, x, :], sub_shape)
                subsum = CP.cp.reshape(CP.cp.sum(submask, axis=(1, 2)), sub_shape)
                target += (subgrad / subsum) * submask

        if ph != 0 or pw != 0:
            result = result[:, ph:height - ph, pw:width - pw, :]

        return result

    def get_output_shapes(self, input_shapes):
        input_shapes = make_list_if_not(input_shapes)
        batch_size, height, width, channels = input_shapes[0]

        kh, kw = self.kernel_size
        ph, pw = self.padding
        sh, sw = self.stride
        ceil = math.ceil if self.ceil_mode else math.floor

        out_height = ceil((height + 2 * ph - (kh - 1) - 1) / sh + 1)
        out_width = ceil((width + 2 * pw - (kw - 1) - 1) / sw + 1)

        return [(batch_size, out_height, out_width, channels)]


class Noop(BaseLayer):
    def _forward(self, X, mem_id=0):
        return X

    def _backward(self, grad, mem_id=0):
        return grad

    def get_output_shapes(self, input_shapes):
        return make_list_if_not(input_shapes)


class Relu(BaseLayer):
    def _forward(self, X, mem_id=0):
        self._mem[mem_id] = X >= 0
        return X * self._mem[mem_id]

    def _backward(self, grad, mem_id=0):
        result = grad * self._mem[mem_id]
        return result

    def get_output_shapes(self, input_shapes):
        return make_list_if_not(input_shapes)


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

        result = CP.cp.zeros(self._mem[mem_id])
        for y, gy in enumerate(range(0, height, sf_y)):
            for x, gx in enumerate(range(0, width, sf_x)):
                region = grad[:, gy:gy + sf_y, gx:gx + sf_x, :]
                result[:, y, x, :] += CP.cp.sum(region, axis=(1, 2))

        return result

    def get_output_shapes(self, input_shapes):
        input_shapes = make_list_if_not(input_shapes)
        return [tuple((input_shapes[0][0],
                       *(np.array(input_shapes[0][1:]) * (*self.scale_factor, 1))))]
