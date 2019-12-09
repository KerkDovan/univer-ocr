from datetime import datetime as dt

import numpy as np

from ..layers import Convolutional2D, MaxPool2D, Upsample2D
from ..gpu import CP


def make_conv_layers(kernel_size, in_channels, out_channels):
    return {
        'Convolutional 2D Layer': Convolutional2D(
            kernel_size, in_channels, out_channels),

        'Convolutional 2D Layer with Padding': Convolutional2D(
            kernel_size, in_channels, out_channels, padding=1),

        'Convolutional 2D Layer with non-zero Padding': Convolutional2D(
            kernel_size, in_channels, out_channels, padding=1, padding_value=0.5),

        'Convolutional 2D Layer with Stride': Convolutional2D(
            kernel_size, in_channels, out_channels, stride=2),

        'Convolutional 2D Layer with Padding and Stride': Convolutional2D(
            kernel_size, in_channels, out_channels, padding=1, stride=2),
    }


def make_pool_layers(kernel_size):
    return {
        'Max Pooling 2D Layer': MaxPool2D(
            kernel_size),

        'Max Pooling 2D Layer with Padding': MaxPool2D(
            kernel_size, padding=1),

        'Max Pooling 2D Layer with 1-Stride': MaxPool2D(
            kernel_size, stride=1),

        'Max Pooling 2D Layer with Padding and 1-Stride': MaxPool2D(
            kernel_size, padding=1, stride=1),
    }


def make_upsample_layers(scale_factor):
    return {
        'Upsampling 2D Layer': Upsample2D(
            scale_factor)
    }


def make_layers(conv_ks, conv_in, conv_out, pool_ks, upsample_sf):
    layers = {
        **make_conv_layers(conv_ks, conv_in, conv_out),
        **make_pool_layers(pool_ks),
        **make_upsample_layers(upsample_sf),
    }
    longest_name_length = max(len(name) for name in layers.keys())
    result = {
        name.ljust(longest_name_length): layer
        for name, layer in layers.items()
    }
    return result


def run_layers(layers, inputs, outputs):
    for name, layer in layers.items():
        X, grad = inputs[name]
        print(f'  {name}... ', end='')
        ts = dt.now()
        y = layer.forward(X)[0]
        dX = layer.backward(grad)[0]
        print(f'done in {dt.now() - ts}')
        outputs[name] = (y, dX)


def main(*args, **kwargs):
    ts_start = dt.now()

    print('Running in CPU mode')

    batch_size = 5
    kernel_size = (3, 3)
    h, w = 240, 320
    in_channels, out_channels = 6, 7
    cpu_layers = make_layers(kernel_size, in_channels, out_channels, (2, 2), (2, 2))
    cpu_inputs, cpu_outputs = {}, {}
    for name, layer in cpu_layers.items():
        X = np.random.randn(batch_size, h, w, in_channels)
        _, out_h, out_w, out_ch = layer.get_output_shapes(X.shape)[0]
        grad = np.random.randn(batch_size, out_h, out_w, out_ch)
        cpu_inputs[name] = X, grad
    run_layers(cpu_layers, cpu_inputs, cpu_outputs)

    print('\nRunning in GPU mode')

    CP.use_gpu()
    gpu_layers = make_layers(kernel_size, in_channels, out_channels, (2, 2), (2, 2))
    for name, cpu_layer in cpu_layers.items():
        gpu_layer = gpu_layers[name]
        gpu_layer.set_weights(cpu_layer.get_weights())

    gpu_inputs = {
        name: (CP.copy(X), CP.copy(grad))
        for name, (X, grad) in cpu_inputs.items()
    }
    gpu_outputs = {}
    run_layers(gpu_layers, gpu_inputs, gpu_outputs)

    print('\nComparing results')
    correct_cnt = 0
    total_cnt = len(cpu_layers)

    for name in cpu_layers.keys():
        X, grad = cpu_inputs[name]
        cpu_y, cpu_dX = cpu_outputs[name]
        gpu_y, gpu_dX = gpu_outputs[name]
        gpu_y = CP.asnumpy(gpu_y)
        gpu_dX = CP.asnumpy(gpu_dX)

        assert type(cpu_y) == type(gpu_y), f'{type(cpu_y)} != {type(gpu_y)}'
        assert type(cpu_dX) == type(gpu_dX), f'{type(cpu_dX)} != {type(gpu_dX)}'

        assert cpu_y.shape == gpu_y.shape, f'{cpu_y.shape} != {gpu_y.shape}'
        assert cpu_dX.shape == gpu_dX.shape, f'{cpu_dX.shape} != {gpu_dX.shape}'

        y_isclose = np.isclose(cpu_y, gpu_y).all()
        dX_isclose = np.isclose(cpu_dX, gpu_dX).all()
        y_isclose_str = str(y_isclose).rjust(5)
        dX_isclose_str = str(dX_isclose).rjust(5)

        correct_cnt += y_isclose and dX_isclose
        print(f'  {name}: y={y_isclose_str}, dX={dX_isclose_str}')

    print(f'\nCorrect: {correct_cnt}/{total_cnt}\nTotal time: {dt.now() - ts_start}')


if __name__ == '__main__':
    main()
