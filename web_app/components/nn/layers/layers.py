import numpy as np

from ..gpu import CP
from ..help_func import make_list_if_not
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
        if not self.trainable:
            return
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
            cur_weights = weights.get(name, None)
            if cur_weights is None:
                continue
            cur_weights = np.array(cur_weights)
            error = None
            if np.any(np.isnan(cur_weights)):
                error = 'NaN found in loaded weights'
            elif cur_weights.shape != param.value.shape:
                error = f'Shapes don`t match: {cur_weights.shape} != {param.value.shape}'
            if error is not None:
                print(f'{self.name}/{name}: {error}, skipping')
                continue
            param.value = CP.copy(cur_weights)

    def nan_weights(self):
        return any(CP.cp.isnan(param.value).any() for param in self.params().values())

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


class BaseLayerGPU(BaseLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _init_forward_backward(self):
        self._forward_cpu = self._make_forward_cpu()
        self._backward_cpu = self._make_backward_cpu()
        self._forward_gpu = self._make_forward_gpu()
        self._backward_gpu = self._make_backward_gpu()

    @track_this('forward')
    def forward(self, inputs):
        assert self.is_initialized, 'You must initialize() layer before calling forward() method'
        inputs = make_list_if_not(inputs)
        result = []
        _forward = self._forward_gpu if CP.is_gpu_used else self._forward_cpu
        for mem_id, X in enumerate(inputs):
            result.append(_forward(self, X, mem_id))
        return result

    @track_this('backward')
    def backward(self, grads):
        grads = make_list_if_not(grads)
        result = []
        _backward = self._backward_gpu if CP.is_gpu_used else self._backward_cpu
        for mem_id, grad in enumerate(grads):
            result.append(_backward(self, grad, mem_id))
        self.clear_memory()
        return result

    def get_kernel_dims(self, array, axis):
        if isinstance(axis, int):
            axis = [axis]
        block_dim = tuple(16 for _ in range(len(axis)))
        grid_dim = tuple(int(np.ceil(array.shape[axis[i]] / block_dim[i]))
                         for i in range(len(axis)))
        return grid_dim, block_dim

    def clear_memory(self):
        raise NotImplementedError()

    def _make_forward_cpu(self):
        raise NotImplementedError()

    def _make_backward_cpu(self):
        raise NotImplementedError()

    def _make_forward_gpu(self):
        raise NotImplementedError()

    def _make_backward_gpu(self):
        raise NotImplementedError()


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


class LeakyRelu(BaseLayer):
    def __init__(self, alpha=0.01, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha

    def _forward(self, X, mem_id=0):
        self._mem[mem_id] = (X >= 0) + self.alpha * (X < 0)
        return X * self._mem[mem_id]

    def _backward(self, grad, mem_id=0):
        result = grad * self._mem[mem_id]
        return result

    def get_output_shapes(self, input_shapes):
        return make_list_if_not(input_shapes)


class Sigmoid(BaseLayer):
    def _forward(self, X, mem_id=0):
        self._mem[mem_id] = X
        return 1 / (1 + CP.cp.exp(-X))

    def _backward(self, grad, mem_id=0):
        X = self._mem[mem_id]
        eX = CP.cp.exp(-X)
        return grad * eX / (eX + 1) ** 2

    def get_output_shapes(self, input_shapes):
        return make_list_if_not(input_shapes)
