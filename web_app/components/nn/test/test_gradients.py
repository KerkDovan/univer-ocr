import datetime
from functools import wraps
from itertools import cycle

import numpy as np

from .. import gradient_check as grad_check
from ..layers import Convolutional2D, Flatten, FullyConnected, Input, MaxPool2D, Relu, Upsample2D
from ..losses import SigmoidCrossEntropy, SoftmaxCrossEntropy
from ..models import LayerConstructor, ModelConstructor, Sequential
from ..regularizations import L1, L2

correct_cnt = 0
total_cnt = 0
total_time = datetime.timedelta(0)


def now():
    return datetime.datetime.now()


def time_it(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        ts = now()
        result = func(*args, **kwargs)
        dt = now() - ts
        print(f'{"Passed" if result else "Error"} {dt}\n')
        global correct_cnt, total_cnt, total_time
        correct_cnt += result
        total_cnt += 1
        total_time += dt
    return wrapper


@time_it
def check_gradient(func, X, delta=1e-5, tol=1e-4):
    return grad_check.check_gradient(func, X, delta=delta, tol=tol)


@time_it
def check_layer(layer, X, param=None, delta=1e-5, tol=1e-4):
    result = None
    if param is None:
        result = grad_check.check_layer_gradient(layer, X, delta=delta, tol=tol)
    else:
        result = grad_check.check_layer_param_gradient(layer, X, param, delta=delta, tol=tol)
    return result


@time_it
def check_model(model, X, y, delta=1e-5, tol=1e-4):
    return grad_check.check_model_gradient(model, X, y, delta=delta, tol=tol)


def main():
    batch_size, n_input, n_output = 3, 2, 5

    X_fc = np.random.randn(batch_size, n_input)
    X_fl = np.random.randn(batch_size, n_input, n_output)

    print('Input Layer (N,)')
    check_layer(Input((..., n_input)), X_fc)

    print('Input Layer (N, M)')
    check_layer(Input((..., n_input, n_output)), X_fl)

    layer = FullyConnected(n_input, n_output)
    print(f'Fully Connected Layer: {layer.count_parameters("w")} parameters')
    check_layer(layer, X_fc)

    print('Fully Connected Layer - Param')
    check_layer(FullyConnected(n_input, n_output), X_fc, 'w')

    print(f'Flatten Layer: {Flatten().count_parameters()} parameters')
    check_layer(Flatten(), X_fl)

    print('L1 Regularization')
    check_gradient(L1(0.1), X_fc)

    print('L2 Regularization')
    check_gradient(L2(0.1), X_fc)

    n_sizes = [4, 7, 5, 3]
    X = np.random.randn(batch_size, n_sizes[0])

    regularizers = [reg(np.random.rand(1)) for i, reg
                    in zip(range(len(n_sizes) - 1), cycle([L1, L2]))]
    layers = [FullyConnected(n_sizes[i], n_sizes[i + 1])
              for i in range(len(n_sizes) - 1)]
    layers_reg = [FullyConnected(n_sizes[i], n_sizes[i + 1], regularizer=regularizers[i])
                  for i in range(len(n_sizes) - 1)]

    model = Sequential(layers, loss=SoftmaxCrossEntropy())
    print(f'Sequential model with Softmax CE Loss: {model.count_parameters()} parameters')
    y = np.array([np.roll([1] + [0] * (n_sizes[-1] - 1), np.random.randint(n_sizes[-1]))
                  for i in range(batch_size)])
    check_model(model, X, y)

    print('Sequential model with Softmax CE Loss and regularization', regularizers)
    model = Sequential(layers_reg, loss=SoftmaxCrossEntropy())
    check_model(model, X, y)

    print('Sequential model with Sigmoid CE Loss')
    y = np.array(np.random.choice([0, 1], size=(batch_size, n_sizes[-1])))
    model = Sequential(layers, loss=SigmoidCrossEntropy())
    check_model(model, X, y)

    print('Sequential model with Sigmoid CE Loss and regularization', regularizers)
    model = Sequential(layers_reg, loss=SigmoidCrossEntropy())
    check_model(model, X, y)

    in_ch, out_ch = 6, 7
    h, w = 5, 5
    ks = (4, 4)
    X_conv2d = np.random.randn(batch_size, h, w, in_ch)

    layer = Convolutional2D(ks, in_ch, out_ch)
    print(f'Convolutional 2D Layer: {layer.count_parameters()} parameters')
    check_layer(layer, X_conv2d)

    print('Convolutional 2D Layer with Padding')
    check_layer(Convolutional2D(ks, in_ch, out_ch, padding=1), X_conv2d)

    print('Convolutional 2D Layer with non-zero Padding')
    check_layer(Convolutional2D(ks, in_ch, out_ch, padding=1, padding_value=0.5), X_conv2d)

    print('Convolutional 2D Layer with Stride')
    check_layer(Convolutional2D(ks, in_ch, out_ch, stride=2), X_conv2d)

    print('Convolutional 2D Layer with Padding and Stride')
    check_layer(Convolutional2D(ks, in_ch, out_ch, padding=1, stride=2), X_conv2d)

    print('Max Pooling 2D Layer')
    X_maxpool2d = np.reshape(np.array([
        [1, 0, 1, 2],
        [0, -1, -1, -1],
        [-1, -1, 1, -2],
    ]), (1, 3, 4, 1))
    print(X_maxpool2d[0, :, :, 0])
    print(MaxPool2D(2, ceil_mode=True).forward(X_maxpool2d)[0, :, :, 0])
    check_layer(MaxPool2D(2), X_conv2d)

    print('Upsamping 2D Layer')
    X_upsample2d = np.reshape(np.array([
        [1, 2],
        [3, 4],
    ], dtype=np.float), (1, 2, 2, 1))
    print(X_upsample2d[0, :, :, 0])
    print(Upsample2D((2, 3)).forward(X_upsample2d)[0, :, :, 0])
    check_layer(Upsample2D(5), X_upsample2d)

    X_conv2d = np.random.randn(batch_size, 7, 7, 3)
    y_conv2d = np.array([np.roll([1] + [0] * 9, np.random.randint(10))
                         for i in range(batch_size)])
    constructor = ModelConstructor(Sequential, loss=SoftmaxCrossEntropy())
    constructor.add([
        LayerConstructor(Convolutional2D, (3, 3), out_channels=4, regularizer=L1(0.1)),
        MaxPool2D(2),
        LayerConstructor(Convolutional2D, (3, 3), out_channels=5, padding=1),
        LayerConstructor(MaxPool2D, 2),
        LayerConstructor(Convolutional2D, (3, 3), out_channels=6, padding=1, padding_value=0.5),
        Relu,
        LayerConstructor(Convolutional2D, (3, 3), out_channels=7, stride=2, regularizer=L2(0.1)),
        Upsample2D(2),
        Relu(),
        LayerConstructor(Flatten),
        LayerConstructor(FullyConnected, n_output=10)
    ])
    model = constructor.construct(input_shape=X_conv2d.shape)

    print(f'Sequential model with Convolutional 2D Layers from Constructors: '
          f'{model.count_parameters()} parameters')
    check_model(model, X_conv2d, y_conv2d)

    print(f'Correct: {correct_cnt}/{total_cnt}\nTotal time: {total_time}')


if __name__ == '__main__':
    main()
