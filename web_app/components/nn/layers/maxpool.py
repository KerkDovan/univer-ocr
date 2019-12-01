import math

from numba import cuda

from ..gpu import CP
from ..help_func import make_list_if_not, tuplize
from .layers import BaseLayerGPU


class MaxPool2D(BaseLayerGPU):
    def __init__(self, kernel_size, padding=0, stride=None, ceil_mode=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kernel_size = tuplize('kernel_size', kernel_size, 2)
        self.padding = tuplize('padding', padding, 2)
        self.stride = self.kernel_size if stride is None else tuplize('stride', stride)
        self.ceil_mode = ceil_mode
        self._init_forward_backward()

    def _make_forward_cpu(self):
        def _forward_cpu(self, X, mem_id=0):
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

        return _forward_cpu

    def _make_backward_cpu(self):
        def _backward_cpu(self, grad, mem_id=0):
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

        return _backward_cpu

    def _make_forward_gpu(self):
        kh, kw = self.kernel_size
        ph, pw = self.padding
        sh, sw = self.stride

        @cuda.jit(device=True)
        def _forward_gpu_kernel_func(X, result, mask, batch, y, x, channel):
            ty, tx = y * sh - ph, x * sw - pw
            max_val = None
            for ky in range(kh):
                cy = ty + ky
                if not 0 <= cy < X.shape[1]:
                    continue
                for kx in range(kw):
                    cx = tx + kx
                    if not 0 <= cx < X.shape[2]:
                        continue
                    cur_val = X[batch, cy, cx, channel]
                    if max_val is None or max_val < cur_val:
                        max_val = cur_val
            if max_val is None:
                return
            my, mx = y * kh, x * kw
            for ky in range(kh):
                cy = ty + ky
                if not 0 <= cy < X.shape[1]:
                    continue
                for kx in range(kw):
                    cx = tx + kx
                    if not 0 <= cx < X.shape[2]:
                        continue
                    if abs(max_val - X[batch, cy, cx, channel]) < 1e-8:
                        mask[batch, my + ky, mx + kx, channel] = True
            result[batch, y, x, channel] = max_val

        @cuda.jit
        def _forward_gpu_kernel(X, result, mask):
            startY, startX = cuda.grid(2)
            gridY = cuda.gridDim.y * cuda.blockDim.y
            gridX = cuda.gridDim.x * cuda.blockDim.x
            for y in range(startY, result.shape[1], gridY):
                for x in range(startX, result.shape[2], gridX):
                    for batch in range(result.shape[0]):
                        for channel in range(result.shape[3]):
                            _forward_gpu_kernel_func(
                                X, result, mask, batch, y, x, channel)

        def _forward_gpu(self, X, mem_id=0):
            output_shape = self.get_output_shapes(X.shape)[0]
            batch_size, out_height, out_width, channels = output_shape
            mask_shape = (batch_size, kh * out_height, kw * out_width, channels)
            result = CP.cp.zeros(output_shape)
            mask = CP.cp.zeros(mask_shape, dtype=CP.cp.bool)
            grid_dim, block_dim = (32, 32), (8, 8)
            _forward_gpu_kernel[grid_dim, block_dim](X, result, mask)
            self._mem[mem_id] = mask, X.shape
            return result

        return _forward_gpu

    def _make_backward_gpu(self):
        kh, kw = self.kernel_size
        ph, pw = self.padding
        sh, sw = self.stride

        @cuda.jit(device=True)
        def _backward_gpu_kernel_func(grad, dx_total, mask, batch, y, x, channel):
            ty, tx = y * sh - ph, x * sw - pw
            my, mx = y * kh, x * kw
            cnt = 0
            for ky in range(kh):
                cy = ty + ky
                if not 0 <= cy < dx_total.shape[1]:
                    continue
                for kx in range(kw):
                    cx = tx + kx
                    if not 0 <= cx < dx_total.shape[2]:
                        continue
                    cnt += mask[batch, my + ky, mx + kx, channel] is True
            if cnt == 0:
                return
            for ky in range(kh):
                cy = ty + ky
                if not 0 <= cy < dx_total.shape[1]:
                    continue
                for kx in range(kw):
                    cx = tx + kx
                    if not 0 <= cx < dx_total.shape[2]:
                        continue
                    cur_grad = grad[batch, y, x, channel] / cnt
                    if mask[batch, my + ky, mx + kx, channel] is True:
                        dx_total[batch, cy, cx, channel] += cur_grad

        @cuda.jit
        def _backward_gpu_kernel(grad, dx_total, mask):
            startY, startX = cuda.grid(2)
            gridY = cuda.gridDim.y * cuda.blockDim.y
            gridX = cuda.gridDim.x * cuda.blockDim.x
            for y in range(startY, grad.shape[1], gridY):
                for x in range(startX, grad.shape[2], gridX):
                    for batch in range(grad.shape[0]):
                        for out_ch in range(grad.shape[3]):
                            _backward_gpu_kernel_func(
                                grad, dx_total, mask, batch, y, x, out_ch)

        def _backward_gpu(self, grad, mem_id=0):
            mask, input_shape = self._mem[mem_id]
            grid_dim, block_dim = (32, 32), (8, 8)
            dx_total = CP.cp.zeros(input_shape)
            _backward_gpu_kernel[grid_dim, block_dim](grad, dx_total, mask)
            return dx_total

        return _backward_gpu

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
