import numpy as np

import cupy as cp

from .gpu import CP


def check_gradient(f, x, delta=1e-5, tol=1e-4):
    """
    Checks the implementation of analytical gradient by comparing
    it to numerical gradient using two-point formula
    Arguments:
        f: function that receives x and computes value and gradient
        x: np array, initial point where gradient is checked
        delta: step to compute numerical gradient
        tol: tolerance for comparing numerical and analytical gradient
    Return:
        bool indicating whether gradients match or not
    """
    assert isinstance(x, CP.cp.ndarray), (
        f'{CP.cp.ndarray.__name__} expected, {type(x).__name__} found')
    assert x.dtype == CP.cp.float, (
        f'{CP.cp.float.__name__} expected, {x.dtype.__name__}')

    fx, analytic_grad = f(x)
    if isinstance(analytic_grad, list):
        analytic_grad = analytic_grad[0]
    analytic_grad = analytic_grad.copy()

    assert analytic_grad.shape == x.shape, f'{analytic_grad.shape} != {x.shape}'

    it = np.nditer(cp.asnumpy(x), flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index
        analytic_grad_at_ix = analytic_grad[ix]
        numeric_grad_at_ix = 0

        d = CP.cp.zeros_like(x)
        d[ix] = delta
        a = f(x + d)[0]
        b = f(x - d)[0]
        numeric_grad_at_ix = (a - b) / (2 * delta)

        if not CP.cp.isclose(numeric_grad_at_ix, analytic_grad_at_ix, tol).all():
            print(f'Gradients are different at {ix}.\n'
                  f'  Analytic: {analytic_grad_at_ix},\n'
                  f'  Numeric: {numeric_grad_at_ix},\n'
                  f'  diff: {abs(analytic_grad_at_ix - numeric_grad_at_ix)},\n'
                  f'  ratio: {analytic_grad_at_ix / numeric_grad_at_ix}')
            return False

        it.iternext()

    return True


def check_layer_gradient(layer, x, delta=1e-5, tol=1e-4):
    """
    Checks gradient correctness for the input and output of a layer
    Arguments:
        layer: neural network layer, with forward and backward functions
        x: starting point for layer input
        delta: step to compute numerical gradient
        tol: tolerance for comparing numerical and analytical gradient
    Returns:
        bool indicating whether gradients match or not
    """
    output = layer.forward(x)
    if isinstance(output, list):
        output = output[0]
    output_weight = CP.cp.random.randn(*output.shape)

    def helper_func(x):
        output = layer.forward(x)
        if isinstance(output, list):
            output = output[0]
        loss = CP.cp.sum(output * output_weight)
        d_out = CP.cp.ones_like(output) * output_weight
        grad = layer.backward(d_out)
        return loss, grad

    return check_gradient(helper_func, x, delta, tol)


def check_layer_param_gradient(layer, x,
                               param_name,
                               delta=1e-5, tol=1e-4):
    """
    Checks gradient correctness for the parameter of the layer
    Arguments:
        layer: neural network layer, with forward and backward functions
        x: starting point for layer input
        param_name: name of the parameter
        delta: step to compute numerical gradient
        tol: tolerance for comparing numerical and analytical gradient
    Returns:
        bool indicating whether gradients match or not
    """
    param = layer.params()[param_name]
    initial_w = param.value

    layer.clear_grads()
    output = layer.forward(x)
    if isinstance(output, list):
        output = output[0]
    output_weight = CP.cp.random.randn(*output.shape)

    def helper_func(w):
        param.value = w
        layer.clear_grads()
        output = layer.forward(x)
        if isinstance(output, list):
            output = output[0]
        loss = CP.cp.sum(output * output_weight)
        d_out = CP.cp.ones_like(output) * output_weight
        layer.backward(d_out)
        grad = param.grad
        return loss, grad

    return check_gradient(helper_func, initial_w, delta, tol)


def check_model_gradient(model, X, y,
                         delta=1e-5, tol=1e-4,
                         check_inputs=False):
    """
    Checks gradient correctness for all model parameters
    Arguments:
        model: neural network model with compute_loss_and_gradients
        X: batch of input data
        y: batch of labels
        delta: step to compute numerical gradient
        tol: tolerance for comparing numerical and analytical gradient
    Returns:
        bool indicating whether gradients match or not
    """

    if not isinstance(X, list):
        X = [X]

    if check_inputs:
        for input_key in range(len(X)):
            print(f'Checking gradient for model input #{input_key}')

            def helper_func(x):
                this_X = [CP.cp.copy(tX) for tX in X]
                this_X[input_key] += x
                loss = model.compute_loss_and_gradients(this_X, y)
                out_loss = loss['output_losses']
                reg_loss = loss['regularization_loss']
                loss = np.sum(out_loss) + reg_loss
                input_grads = model.input_grads[input_key]
                if isinstance(input_grads, list):
                    input_grads = input_grads[0]
                return loss, input_grads

            zero_X = CP.cp.zeros_like(X[input_key])

            if not check_gradient(helper_func, zero_X, delta, tol):
                return False

    params = model.params()

    for param_key in params:
        print(f'Checking gradient for {param_key}')
        param = params[param_key]
        initial_w = param.value

        def helper_func(w):
            param.value = w
            loss = model.compute_loss_and_gradients(X, y)
            out_loss = loss['output_losses']
            reg_loss = loss['regularization_loss']
            loss = np.sum(out_loss) + reg_loss
            grad = param.grad
            return loss, grad

        if not check_gradient(helper_func, initial_w, delta, tol):
            return False

    return True
