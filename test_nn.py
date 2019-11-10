import datetime
from functools import wraps
from itertools import cycle

import numpy as np

import nn.gradient_check as grad_check
from nn.layers import Flatten, FullyConnected
from nn.losses import SigmoidCrossEntropy, SoftmaxCrossEntropy
from nn.models import Sequential
from nn.regularizations import L1, L2


def now():
    return datetime.datetime.now()


def time_it(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        ts = now()
        result = func(*args, **kwargs)
        print(result, f'{now() - ts}', '\n')
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

    fc = FullyConnected(n_input, n_output)
    fl = Flatten()

    print('Fully Connected Layer')
    check_layer(fc, X_fc)
    fc.clear_grads()

    print('Fully Connected Layer - Param')
    check_layer(fc, X_fc, 'w')

    print('Flatten Layer')
    check_layer(fl, X_fl)

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
    layers_reg = [FullyConnected(n_sizes[i], n_sizes[i + 1], reg=regularizers[i])
                  for i in range(len(n_sizes) - 1)]

    print('Sequential model with Softmax CE Loss')
    y = np.array([np.roll([1] + [0] * (n_sizes[-1] - 1), np.random.randint(n_sizes[-1]))
                  for i in range(batch_size)])
    model = Sequential(layers, SoftmaxCrossEntropy())
    check_model(model, X, y)

    print('Sequential model with Softmax CE Loss and regularization', regularizers)
    model = Sequential(layers_reg, SoftmaxCrossEntropy())
    check_model(model, X, y)

    print('Sequential model with Sigmoid CE Loss')
    y = np.array(np.random.choice([0, 1], size=(batch_size, n_sizes[-1])))
    model = Sequential(layers, SigmoidCrossEntropy())
    check_model(model, X, y)

    print('Sequential model with Sigmoid CE Loss and regularization', regularizers)
    model = Sequential(layers_reg, SigmoidCrossEntropy())
    check_model(model, X, y)


if __name__ == '__main__':
    main()
