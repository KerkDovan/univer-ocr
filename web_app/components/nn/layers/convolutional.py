import math

import numpy as np

from numba import cuda

from ..gpu import CP
from ..help_func import make_list_if_not, tuplize
from .layers import BaseLayer, BaseLayerGPU, Param


class Convolutional2D(BaseLayerGPU):
    def __init__(self, kernel_size, in_channels=None, out_channels=None,
                 padding=0, padding_value=0, stride=1,
                 w=None, b=None, bias=True, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.kernel_size = tuplize('kernel_size', kernel_size, 2)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.padding = tuplize('padding', padding, 2)
        self.padding_value = padding_value
        self.stride = tuplize('stride', stride, 2)
        self.w, self.b, self.bias = w, b, bias

        if self.input_shapes is None and in_channels is not None:
            self.input_shapes = [(None, None, None, self.in_channels)]
        if self.input_shapes is not None:
            self.initialize(self.input_shapes)
        else:
            self.is_initialized = False

        self._init_forward_backward()

    def initialize(self, input_shapes):
        self.input_shapes = input_shapes
        self.in_channels = self.input_shapes[0][3]
        if self.out_channels is None:
            self.out_channels = self.in_channels

        w_shape = (*self.kernel_size, self.in_channels, self.out_channels)
        b_shape = (self.out_channels,)
        wb_init_shape = (np.prod(w_shape[:3]) + 1, self.out_channels)

        wb = self.initializer(*wb_init_shape)
        w = CP.cp.reshape(wb[:-1, :], w_shape) if self.w is None else self.w
        b = CP.cp.reshape(wb[-1, :], b_shape) if self.b is None else self.b
        assert w.shape == w_shape, f'{w.shape} != {w_shape}'
        assert b.shape == b_shape, f'{b.shape} != {b_shape}'

        self.w = Param(CP.cp.array(w), optimizer=self.optimizer)
        self.b = Param(CP.cp.array(b), optimizer=self.optimizer)

        self._init_optimizer()
        self.is_initialized = True

    def clear_memory(self):
        for X in self._mem.items():
            del X
        self._mem = {}

    def _make_forward_cpu(self):
        def _forward_cpu(self, X, mem_id=0):
            batch_size, height, width, channels = X.shape
            assert channels == self.in_channels

            kh, kw = self.kernel_size
            ph, pw = self.padding
            sh, sw = self.stride

            w_shape = (np.prod(self.w.value.shape[:3]), self.w.value.shape[3])
            w = CP.cp.reshape(self.w.value, w_shape)
            w = CP.cp.concatenate((w, CP.cp.reshape(self.b.value, (1, *self.b.value.shape))))

            output_shape = self.get_output_shapes(X.shape)[0]
            _, out_height, out_width, _ = output_shape

            if ph != 0 or pw != 0:
                nh, nw = height + 2 * ph, width + 2 * pw
                padded = self.padding_value * CP.cp.ones((batch_size, nh, nw, channels))
                padded[:, ph:nh - ph, pw:nw - pw, :] = X
                X = padded

            self._mem[mem_id] = X

            result = CP.cp.zeros(output_shape)
            bias_vec = self.bias * CP.cp.ones((batch_size, 1))
            new_shape = (batch_size, w.shape[0] - 1)

            for y in range(0, out_height):
                for x in range(0, out_width):
                    in_y, in_x = y * sh, x * sw
                    i = CP.cp.reshape(X[:, in_y:in_y + kh, in_x:in_x + kw, :], new_shape)
                    i = CP.cp.concatenate((i, bias_vec), axis=1)
                    res = CP.cp.dot(i, w)
                    result[:, y, x, :] = res

            return result
        return _forward_cpu

    def _make_backward_cpu(self):
        def _backward_cpu(self, grad, mem_id=0):
            X = self._mem[mem_id]
            batch_size, height, width, in_channels = X.shape
            _, out_height, out_width, out_channels = grad.shape

            kh, kw = self.kernel_size
            ph, pw = self.padding
            sh, sw = self.stride

            w_shape = (np.prod(self.w.value.shape[:3]), self.w.value.shape[3])
            w = CP.cp.reshape(self.w.value, w_shape)
            w = CP.cp.concatenate((w, CP.cp.reshape(self.b.value, (1, *self.b.value.shape))))

            bias_vec = self.bias * CP.cp.ones((batch_size, 1))
            new_shape = (batch_size, w_shape[0])
            dx_total = CP.cp.zeros_like(X)
            dw_temp = CP.cp.zeros_like(w)
            wt = CP.cp.transpose(w[:-1, :])

            for y in range(0, out_height):
                for x in range(0, out_width):
                    in_y, in_x = y * sh, x * sw
                    cur_grad = grad[:, y, x, :]

                    xt = CP.cp.reshape(X[:, in_y:in_y + kh, in_x:in_x + kw, :], new_shape)
                    xt = CP.cp.concatenate((xt, bias_vec), axis=1)
                    xt = CP.cp.transpose(xt)
                    dw = CP.cp.dot(xt, cur_grad)
                    dw_temp += dw

                    dx = CP.cp.dot(cur_grad, wt)
                    dx = CP.cp.reshape(dx, (batch_size, kh, kw, in_channels))
                    dx_total[:, in_y:in_y + kh, in_x:in_x + kw, :] += dx

            db_total = dw_temp[-1, :]
            dw_total = CP.cp.reshape(CP.cp.delete(dw_temp, -1, axis=0), self.w.value.shape)
            self.w.grad += dw_total
            self.b.grad += db_total

            if ph != 0 or pw != 0:
                dx_total = dx_total[:, ph:height - ph, pw:width - pw, :]

            return dx_total
        return _backward_cpu

    def _make_forward_gpu(self):
        kh, kw = self.kernel_size
        ph, pw = self.padding
        sh, sw = self.stride
        padding_value = self.padding_value

        @cuda.jit(device=True)
        def _forward_gpu_kernel_func(X, w, b, result, batch, y, x, out_ch):
            ty, tx = y * sh - ph, x * sw - pw
            for ky in range(kh):
                cy = ty + ky
                for kx in range(kw):
                    cx = tx + kx
                    is_out_of_bounds = False
                    if not 0 <= cy < X.shape[1] or not 0 <= cx < X.shape[2]:
                        is_out_of_bounds = True
                    for in_ch in range(X.shape[3]):
                        cur_X = padding_value if is_out_of_bounds else X[batch, cy, cx, in_ch]
                        cur_w = w[ky, kx, in_ch, out_ch]
                        res = cur_w * cur_X
                        result[batch, y, x, out_ch] += res
            cur_b = b[out_ch]
            result[batch, y, x, out_ch] += cur_b

        @cuda.jit
        def _forward_gpu_kernel(X, w, b, result):
            startX, startY, startZ = cuda.grid(3)
            gridX = cuda.gridDim.x * cuda.blockDim.x
            gridY = cuda.gridDim.y * cuda.blockDim.y
            gridZ = cuda.gridDim.z * cuda.blockDim.z
            for y in range(startY, result.shape[1], gridY):
                for x in range(startX, result.shape[2], gridX):
                    for out_ch in range(startZ, result.shape[3], gridZ):
                        for batch in range(result.shape[0]):
                            _forward_gpu_kernel_func(
                                X, w, b, result,
                                batch, y, x, out_ch)

        def _forward_gpu(self, X, mem_id=0):
            self._mem[mem_id] = X
            output_shape = self.get_output_shapes(X.shape)[0]
            result = CP.cp.zeros(output_shape)
            grid_dim, block_dim = self.get_kernel_dims(result, (1, 2, 3))
            _forward_gpu_kernel[grid_dim, block_dim](
                X, self.w.value, self.b.value, result)
            cuda.synchronize()
            return result

        return _forward_gpu

    def _make_backward_gpu(self):
        kh, kw = self.kernel_size
        ph, pw = self.padding
        sh, sw = self.stride
        padding_value = self.padding_value

        @cuda.jit(device=True)
        def _backward_gpu_kernel_dx_func(w, grad, dx_total, batch, y, x, in_ch):
            for ky in range(kh):
                tmp_gy = y - ky + ph
                gy = tmp_gy // sh
                if gy * sh != tmp_gy or not 0 <= gy < grad.shape[1]:
                    continue
                for kx in range(kw):
                    tmp_gx = x - kx + pw
                    gx = tmp_gx // sw
                    if gx * sw != tmp_gx or not 0 <= gx < grad.shape[2]:
                        continue
                    for out_ch in range(grad.shape[3]):
                        cur_grad = grad[batch, gy, gx, out_ch]
                        cur_w = w[ky, kx, in_ch, out_ch]
                        dx = cur_grad * cur_w
                        dx_total[batch, y, x, in_ch] += dx

        @cuda.jit(device=True)
        def _backward_gpu_kernel_dw_db_func(X, grad, dw_total, db_total, batch, y, x, out_ch):
            ty, tx = y * sh - ph, x * sw - pw
            cur_grad = grad[batch, y, x, out_ch]
            for ky in range(kh):
                cy = ty + ky
                for kx in range(kw):
                    cx = tx + kx
                    is_out_of_bounds = False
                    if not 0 <= cy < X.shape[1] or not 0 <= cx < X.shape[2]:
                        is_out_of_bounds = True
                    for in_ch in range(X.shape[3]):
                        cur_X = padding_value if is_out_of_bounds else X[batch, cy, cx, in_ch]
                        dw = cur_grad * cur_X
                        dw_total[ky, kx, in_ch, out_ch] += dw
            db = cur_grad
            db_total[out_ch] += db

        @cuda.jit
        def _backward_gpu_kernel_dx(w, grad, dx_total):
            startX, startY, startZ = cuda.grid(3)
            gridX = cuda.gridDim.x * cuda.blockDim.x
            gridY = cuda.gridDim.y * cuda.blockDim.y
            gridZ = cuda.gridDim.z * cuda.blockDim.z
            for y in range(startY, dx_total.shape[1], gridY):
                for x in range(startX, dx_total.shape[2], gridX):
                    for in_ch in range(startZ, dx_total.shape[3], gridZ):
                        for batch in range(grad.shape[0]):
                            _backward_gpu_kernel_dx_func(
                                w, grad, dx_total, batch, y, x, in_ch)

        @cuda.jit
        def _backward_gpu_kernel_dw_db(X, grad, dw_total, db_total):
            startX, startY, startZ = cuda.grid(3)
            gridX = cuda.gridDim.x * cuda.blockDim.x
            gridY = cuda.gridDim.y * cuda.blockDim.y
            gridZ = cuda.gridDim.z * cuda.blockDim.z
            for y in range(startY, grad.shape[1], gridY):
                for x in range(startX, grad.shape[2], gridX):
                    for out_ch in range(startZ, grad.shape[3], gridZ):
                        for batch in range(grad.shape[0]):
                            _backward_gpu_kernel_dw_db_func(
                                X, grad,
                                dw_total[startY, startX], db_total[startY, startX],
                                batch, y, x, out_ch)

        def _backward_gpu(self, grad, mem_id=0):
            X = self._mem[mem_id]

            grid_dim, block_dim = self.get_kernel_dims(grad, (1, 2, 3))
            dx_total = CP.cp.zeros_like(X)
            _backward_gpu_kernel_dx[grid_dim, block_dim](self.w.value, grad, dx_total)

            dw_total = CP.cp.zeros((16, 16, *self.w.value.shape))
            db_total = CP.cp.zeros((16, 16, *self.b.value.shape))
            grid_dim, block_dim = self.get_kernel_dims(dw_total, (0, 1, -1))
            _backward_gpu_kernel_dw_db[grid_dim, block_dim](X, grad, dw_total, db_total)
            cuda.synchronize()

            dw_total = CP.cp.sum(dw_total, axis=(0, 1))
            db_total = CP.cp.sum(db_total, axis=(0, 1))
            self.w.grad += dw_total
            self.b.grad += db_total
            del dw_total, db_total

            return dx_total

        return _backward_gpu

    def get_output_shapes(self, input_shapes):
        input_shapes = make_list_if_not(input_shapes)
        batch_size, height, width, _ = input_shapes[0]

        kh, kw = self.kernel_size
        ph, pw = self.padding
        sh, sw = self.stride

        out_height = math.floor((height + 2 * ph - (kh - 1) - 1) / sh + 1)
        out_width = math.floor((width + 2 * pw - (kw - 1) - 1) / sw + 1)

        return [(batch_size, out_height, out_width, self.out_channels)]

    def changes_receptive_field(self):
        return True

    def _get_receptive_field(self, axis, position, output_id):
        assert 0 <= axis < 2, f'Convolutional2D has two axis, found {axis}'
        assert output_id < self.get_outputs_count(), (
            f'This layer has only {self.get_outputs_count()} outputs')

        if (axis, position, output_id) in self._receptive_fields:
            return self._receptive_fields[axis, position, output_id]

        k = self.kernel_size[axis]
        p = self.padding[axis]
        s = self.stride[axis]

        result = []
        start = position * s - p
        for ki in range(k):
            result.append(start + ki)

        self._receptive_fields[axis, position, output_id] = {0: set(result)}
        return self._receptive_fields[axis, position, output_id]

    def params(self):
        return {'w': self.w, 'b': self.b}


class Conv2DToBatchedFixedWidthed(BaseLayer):
    def __init__(self, width, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.width = width

    def _forward(self, X, mem_id=0):
        new_shape = self.get_output_shapes(X.shape)[0]
        bs, h, w, ch = X.shape
        hw = self.width // 2
        padded = CP.cp.zeros((bs, h, w + self.width, ch))
        padded[:, :, hw:-self.width + hw, :] = X
        y = CP.cp.zeros(new_shape)
        out_bs = 0
        for in_bs in range(bs):
            for w_id in range(w):
                y[out_bs, :, :, :] = padded[in_bs, :, w_id:w_id + self.width, :]
                out_bs += 1
        self._mem[mem_id] = padded.shape
        return y

    def _backward(self, grad, mem_id=0):
        padded_shape = self._mem[mem_id]
        bs, h, w, ch = padded_shape
        dx = CP.cp.zeros(padded_shape)
        out_bs = 0
        for in_bs in range(bs):
            for w_id in range(w - self.width):
                dx[in_bs, :, w_id:w_id + self.width, :] += grad[out_bs, :, :, :]
                out_bs += 1
        hw = self.width // 2
        return dx[:, :, hw:-self.width + hw, :]

    def get_output_shapes(self, input_shapes):
        input_shapes = make_list_if_not(input_shapes)
        output_shapes = []
        for i in range(len(input_shapes)):
            bs, h, w, ch = input_shapes[i]
            assert w >= self.width, (
                f'Input width must be >= than output width, '
                f'found: {w} < {self.width}')
            new_w = self.width
            new_bs = bs * w
            output_shapes.append((new_bs, h, new_w, ch))
        return output_shapes
