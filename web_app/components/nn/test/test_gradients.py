import datetime
from functools import wraps
from itertools import cycle

import numpy as np

from .. import gradient_check as grad_check
from ..layers import (
    Concat, Convolutional2D, Flatten, FullyConnected, MaxPool2D, Noop, Relu, Upsample2D)
from ..losses import (
    SegmentationDice2D, SegmentationJaccard2D, SigmoidCrossEntropy, SoftmaxCrossEntropy)
from ..models import Model, Sequential
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
    model.initialize_from_X(X)
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
        [0.1, 0.2],
        [0.3, 0.4],
    ], dtype=np.float), (1, 2, 2, 1)).repeat(4, axis=0).repeat(3, axis=-1)
    upsample2d = Upsample2D((2, 3))
    result = upsample2d.forward(X_upsample2d)
    grad = upsample2d.backward(result)
    print(X_upsample2d[0, :, :, 0], result[0, :, :, 0], grad[0, :, :, 0], sep='\n')
    check_layer(Upsample2D(5), X_upsample2d)

    X_dice = np.random.randn(4, 4, 8, 3)
    X_dice -= np.min(X_dice) - np.random.uniform(low=0.0001, high=0.001)
    X_dice /= np.max(X_dice) + np.random.uniform(low=0.0001, high=0.001)
    gt_dice = np.random.randint(0, 2, size=(X_dice.shape[0], 11, 16, 5))
    layers = [
        Convolutional2D((3, 3), X_dice.shape[-1], 2, padding=1),
        Convolutional2D((3, 3), 2, 3, padding=1),
        MaxPool2D(3),
        Convolutional2D((2, 2), 3, 4, padding=1),
        Upsample2D(5),
        Noop(),
        Relu(),
        Convolutional2D((2, 2), 4, gt_dice.shape[-1], padding=1),
    ]

    print(f'Sequential FCN with Segmentation Dice 2D Loss')
    model = Sequential(layers, loss=SegmentationDice2D())
    model.initialize_from_X(X_dice)
    check_model(model, X_dice, gt_dice)

    print(f'Sequential FCN with Segmentation Jaccard 2D Loss')
    model = Sequential(layers, loss=SegmentationJaccard2D())
    check_model(model, X_dice, gt_dice)

    print(f'Concat Layer')
    concat = Concat()
    X_concat = [np.array([[[1, 2, 3]]], dtype=np.float),
                np.array([[[4, 5, 6]]], dtype=np.float)]
    result = concat.forward(X_concat)
    grads = concat.backward(result)
    print(X_concat, result, grads, sep='\n')
    check_layer(concat, X_concat[0])

    batch_size = 5
    in_size, out_size = 1, 3
    in_cnt, out_cnt = 1, 2
    in_shape = (batch_size, 5, 5, in_size)
    out_shape = (batch_size, out_size)
    X = [np.random.randn(*in_shape) for _ in range(in_cnt)]
    y = [np.random.randint(2, size=out_shape) for _ in range(out_cnt)]
    layers = {
        'conv1': Convolutional2D((2, 2), out_channels=out_size),
        'conv2': Convolutional2D((2, 2), out_channels=out_size),
        'concat': Concat(),
        'pool': MaxPool2D(2),
        'flatten': Flatten(),
        'dense1': FullyConnected(n_output=out_shape[1]),
        'dense2': FullyConnected(n_output=out_shape[1]),
    }
    relations = {
        'conv1': 0,
        'conv2': 0,
        'concat': ['conv1', 'conv2'],
        'pool': 'concat',
        'flatten': 'pool',
        'dense1': 'flatten',
        'dense2': 'dense1',
        0: 'dense1',
        1: 'dense2',
    }
    model = Model(layers, relations, loss=SigmoidCrossEntropy())
    model.initialize_from_X(X)
    print(f'Small non-sequential model: {model.count_parameters()} parameters')
    check_model(model, X, y)

    batch_size = 3
    in_channels, out_channels = 3, 3
    in_shape = (batch_size, 18, 18, in_channels)
    out_shape = (batch_size, 1, 1, out_channels)
    X = [np.random.randn(*in_shape), np.random.randn(*in_shape)]
    y = np.random.randint(2, size=out_shape)

    def make_conv_submodel(out_ch):
        return Sequential([
            Convolutional2D((2, 2), out_channels=out_ch),
            Convolutional2D((2, 2), out_channels=out_ch),
            MaxPool2D((2, 2)),
        ])

    layers = {
        'row_1': make_conv_submodel(2),
        'row_2': make_conv_submodel(3),
        'concat_rows': Concat(),

        'concat_inputs': Concat(),
        'row_inputs': make_conv_submodel(2),

        'concat_all': Concat(),

        'pool_1': MaxPool2D((2, 2)),
        'pool_2': MaxPool2D((2, 2)),
        'conv_end': Convolutional2D((2, 2), out_channels=out_channels),
    }
    relations = {
        'row_1': 0,
        'row_2': 1,
        'concat_rows': ['row_1', 'row_2'],

        'concat_inputs': [0, 1],
        'row_inputs': 'concat_inputs',

        'concat_all': ['concat_rows', 'row_inputs'],
        'pool_1': 'concat_all',
        'pool_2': 'pool_1',
        'conv_end': 'pool_2',
        0: 'conv_end',
    }
    model = Model(layers, relations, loss=SegmentationDice2D())
    model.initialize_from_X(X)
    print(f'Big non-sequential model: {model.count_parameters()} parameters')
    check_model(model, X, y)

    print(f'Correct: {correct_cnt}/{total_cnt}\nTotal time: {total_time}')


if __name__ == '__main__':
    main()
