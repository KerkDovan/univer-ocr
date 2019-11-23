import numpy as np


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
    assert isinstance(x, np.ndarray)
    assert x.dtype == np.float

    fx, analytic_grad = f(x)
    analytic_grad = analytic_grad.copy()

    assert analytic_grad.shape == x.shape, f'{analytic_grad.shape} != {x.shape}'

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index
        analytic_grad_at_ix = analytic_grad[ix]
        numeric_grad_at_ix = 0

        d = np.zeros_like(x)
        d[ix] = delta
        a = f(x + d)[0]
        b = f(x - d)[0]
        numeric_grad_at_ix = (a - b) / (2 * delta)

        if not np.isclose(numeric_grad_at_ix, analytic_grad_at_ix, tol).all():
            print(f"Gradients are different at {ix}.\n"
                  f"  Analytic: {analytic_grad_at_ix},\n"
                  f"  Numeric: {numeric_grad_at_ix},\n"
                  f"  diff: {abs(analytic_grad_at_ix - numeric_grad_at_ix)},\n"
                  f"  ratio: {analytic_grad_at_ix / numeric_grad_at_ix}")
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
    output_weight = np.random.randn(*output.shape)

    def helper_func(x):
        output = layer.forward(x)
        loss = np.sum(output * output_weight)
        d_out = np.ones_like(output) * output_weight
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

    output = layer.forward(x)
    output_weight = np.random.randn(*output.shape)

    def helper_func(w):
        param.value = w
        output = layer.forward(x)
        loss = np.sum(output * output_weight)
        d_out = np.ones_like(output) * output_weight
        layer.backward(d_out)
        grad = param.grad
        return loss, grad

    return check_gradient(helper_func, initial_w, delta, tol)


def check_model_gradient(model, X, y,
                         delta=1e-5, tol=1e-4):
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
    params = model.params()

    for param_key in params:
        print("Checking gradient for %s" % param_key)
        param = params[param_key]
        initial_w = param.value

        def helper_func(w):
            param.value = w
            loss = model.compute_loss_and_gradients(X, y)
            loss = np.sum(loss['output_losses']) + loss['regularization_loss']
            grad = param.grad
            return loss, grad

        if not check_gradient(helper_func, initial_w, delta, tol):
            return False

    return True
