import numpy as np

from numba import cuda

from ..gpu import CP
from ..help_func import make_list_if_not, tuplize
from .layers import BaseLayerGPU


class Upsample2D(BaseLayerGPU):
    def __init__(self, scale_factor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scale_factor = tuplize('scale_factor', scale_factor, 2)
        self._init_forward_backward()

    def clear_memory(self):
        for input_shape in self._mem.items():
            del input_shape
        self._mem = {}

    def _make_forward_cpu(self):
        def _forward_cpu(self, X, mem_id=0):
            self._mem[mem_id] = X.shape
            return X.repeat(self.scale_factor[0], axis=1).repeat(self.scale_factor[1], axis=2)
        return _forward_cpu

    def _make_backward_cpu(self):
        def _backward_cpu(self, grad, mem_id=0):
            batch_size, height, width, channels = grad.shape
            sf_y, sf_x = self.scale_factor

            result = CP.cp.zeros(self._mem[mem_id])
            for y, gy in enumerate(range(0, height, sf_y)):
                for x, gx in enumerate(range(0, width, sf_x)):
                    region = grad[:, gy:gy + sf_y, gx:gx + sf_x, :]
                    result[:, y, x, :] += CP.cp.sum(region, axis=(1, 2))

            return result
        return _backward_cpu

    def _make_forward_gpu(self):
        sh, sw = self.scale_factor

        @cuda.jit(device=True)
        def _forward_gpu_kernel_func(X, result, batch, y, x, channel):
            ty, tx = y * sh, x * sw
            for sy in range(sh):
                ry = ty + sy
                for sx in range(sw):
                    rx = tx + sx
                    result[batch, ry, rx, channel] = X[batch, y, x, channel]

        @cuda.jit
        def _forward_gpu_kernel(X, result):
            startX, startY, startZ = cuda.grid(3)
            gridX = cuda.gridDim.x * cuda.blockDim.x
            gridY = cuda.gridDim.y * cuda.blockDim.y
            gridZ = cuda.gridDim.z * cuda.blockDim.z
            for y in range(startY, X.shape[1], gridY):
                for x in range(startX, X.shape[2], gridX):
                    for channel in range(startZ, X.shape[3], gridZ):
                        for batch in range(X.shape[0]):
                            _forward_gpu_kernel_func(
                                X, result, batch, y, x, channel)

        def _forward_gpu(self, X, mem_id=0):
            self._mem[mem_id] = X.shape
            output_shape = self.get_output_shapes(X.shape)[0]
            result = CP.cp.zeros(output_shape)
            grid_dim, block_dim = self.get_kernel_dims(X, (1, 2, 3))
            _forward_gpu_kernel[grid_dim, block_dim](X, result)
            cuda.synchronize()
            return result

        return _forward_gpu

    def _make_backward_gpu(self):
        sh, sw = self.scale_factor

        @cuda.jit(device=True)
        def _backward_gpu_kernel_func(grad, dx_total, batch, y, x, channel):
            ty, tx = y * sh, x * sw
            for sy in range(sh):
                ry = ty + sy
                for sx in range(sw):
                    rx = tx + sx
                    dx_total[batch, y, x, channel] += grad[batch, ry, rx, channel]

        @cuda.jit
        def _backward_gpu_kernel(grad, dx_total):
            startX, startY, startZ = cuda.grid(3)
            gridX = cuda.gridDim.x * cuda.blockDim.x
            gridY = cuda.gridDim.y * cuda.blockDim.y
            gridZ = cuda.gridDim.z * cuda.blockDim.z
            for y in range(startY, dx_total.shape[1], gridY):
                for x in range(startX, dx_total.shape[2], gridX):
                    for channel in range(startZ, dx_total.shape[3], gridZ):
                        for batch in range(dx_total.shape[0]):
                            _backward_gpu_kernel_func(
                                grad, dx_total, batch, y, x, channel)

        def _backward_gpu(self, grad, mem_id=0):
            input_shape = self._mem[mem_id]
            dx_total = CP.cp.zeros(input_shape)
            grid_dim, block_dim = self.get_kernel_dims(grad, (1, 2, 3))
            _backward_gpu_kernel[grid_dim, block_dim](grad, dx_total)
            cuda.synchronize()
            return dx_total

        return _backward_gpu

    def get_output_shapes(self, input_shapes):
        input_shapes = make_list_if_not(input_shapes)
        return [tuple((input_shapes[0][0],
                       *(np.array(input_shapes[0][1:]) * (*self.scale_factor, 1))))]

    def changes_receptive_field(self):
        return True

    def _get_receptive_field(self, axis, position, output_id):
        assert 0 <= axis < 2, f'Upsample2D has two axis, found {axis}'
        assert output_id < self.get_outputs_count(), (
            f'This layer has only {self.get_outputs_count()} outputs')

        if (axis, position, output_id) in self._receptive_fields:
            return self._receptive_fields[axis, position, output_id]

        sf = self.scale_factor[axis]

        result = []
        start = position // sf
        result.append(start)

        self._receptive_fields[axis, position, output_id] = {0: set(result)}
        return self._receptive_fields[axis, position, output_id]
