import datetime
from functools import wraps
from itertools import cycle

import numpy as np

import nn.gradient_check as grad_check
from nn.layers import Convolutional2D, Flatten, FullyConnected
from nn.losses import SigmoidCrossEntropy, SoftmaxCrossEntropy
from nn.models import Sequential
from nn.regularizations import L1, L2

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

    print('Fully Connected Layer')
    check_layer(FullyConnected(n_input, n_output), X_fc)

    print('Fully Connected Layer - Param')
    check_layer(FullyConnected(n_input, n_output), X_fc, 'w')

    print('Flatten Layer')
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

    print('Sequential model with Softmax CE Loss')
    y = np.array([np.roll([1] + [0] * (n_sizes[-1] - 1), np.random.randint(n_sizes[-1]))
                  for i in range(batch_size)])
    model = Sequential(layers, loss=SoftmaxCrossEntropy())
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

    print('Convolutional 2D Layer')
    check_layer(Convolutional2D(ks, in_ch, out_ch), X_conv2d)

    print('Convolutional 2D Layer with Padding')
    check_layer(Convolutional2D(ks, in_ch, out_ch, padding=1), X_conv2d)

    print('Convolutional 2D Layer with non-zero Padding')
    check_layer(Convolutional2D(ks, in_ch, out_ch, padding=1, padding_value=0.5), X_conv2d)

    print('Convolutional 2D Layer with Stride')
    check_layer(Convolutional2D(ks, in_ch, out_ch, stride=2), X_conv2d)

    print('Convolutional 2D Layer with Padding and Stride')
    check_layer(Convolutional2D(ks, in_ch, out_ch, padding=1, stride=2), X_conv2d)

    X_conv2d = np.random.randn(batch_size, 7, 7, 3)
    y_conv2d = np.array([np.roll([1] + [0] * 9, np.random.randint(10))
                         for i in range(batch_size)])
    conv2d_layers = [
        Convolutional2D((3, 3), 3, 4, regularizer=L1(0.1)),
        Convolutional2D((3, 3), 4, 5, padding=1),
        Convolutional2D((3, 3), 5, 6, padding=1, padding_value=0.5),
        Convolutional2D((3, 3), 6, 7, stride=2, regularizer=L2(0.1)),
        Flatten(),
        FullyConnected(28, 10)
    ]
    model = Sequential(conv2d_layers, loss=SoftmaxCrossEntropy())

    print('Sequential model with Convolutional 2D Layers')
    check_model(model, X_conv2d, y_conv2d)

    print(f'Correct: {correct_cnt}/{total_cnt}\nTotal time: {total_time}')


if __name__ == '__main__':
    main()
