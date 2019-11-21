from .layers import BaseLayer, Input
from .losses import SoftmaxCrossEntropy
from .optimizers import Adam


class BaseModel:
    def compute_loss_and_gradients(self, X, y):
        raise NotImplementedError()

    def train(self, X, y):
        loss = self.compute_loss_and_gradients(X, y)
        for param in self.params().values():
            param.update_grad()
            param.clear_grad()
        return loss

    def predict(self, X):
        raise NotImplementedError()

    def params(self):
        raise NotImplementedError()

    def count_parameters(self):
        raise NotImplementedError()


class Sequential(BaseModel):
    def __init__(self, layers, optimizer=Adam(), loss=SoftmaxCrossEntropy()):
        self.layers = layers
        self.optimizer = optimizer
        self.loss = loss

    def compute_loss_and_gradients(self, X, y):
        if not isinstance(X, list):
            X = [X]
        if not isinstance(y, list):
            y = [y]

        pred = X
        for layer in self.layers:
            layer.clear_grads()
            pred = layer.forward(pred)

        loss, grad = self.loss(pred[0], y[0])
        grad = [grad]

        for layer in self.layers[::-1]:
            grad = layer.backward(grad)

        return loss

    def predict(self, X):
        pred = X
        for layer in self.layers:
            pred = layer.forward(pred)
        return pred

    def params(self):
        result = {
            f'layer_{i}_{name}': param
            for i, layer in enumerate(self.layers)
            for name, param in layer.params().items()
        }
        return result

    def count_parameters(self):
        return sum([layer.count_parameters() for layer in self.layers])


class ModelConstructor:
    def __init__(self, model_class, *args, **kwargs):
        self.model_class = model_class
        self.args = args
        self.kwargs = kwargs
        self.layer_constructors = []

    def add(self, layer_constructors):
        if ((isinstance(layer_constructors, LayerConstructor) or
             isinstance(layer_constructors, BaseLayer))):
            self._add_one(layer_constructors)
            return
        for lc in layer_constructors:
            self._add_one(lc)

    def _add_one(self, layer_constructor):
        self.layer_constructors.append(layer_constructor)

    def construct(self, input_shape=None, *args, **kwargs):
        is_input_first = self.layer_constructors[0].layer_class == Input
        if not is_input_first and input_shape is None:
            raise ValueError(f'You must either provide Input layer as the first layer'
                             f'(found: {self.layer_constructors[0].layer_class}) '
                             f'or provide input_shape argument')

        layers = []
        start_from = 1
        if input_shape is not None:
            layer = Input(input_shape=input_shape)
            if not is_input_first:
                start_from = 0
        else:
            layer = self.layer_constructors[0].construct()

        layers.append(layer)

        input_shape = layers[0].input_shape
        for lc in self.layer_constructors[start_from:]:
            if isinstance(lc, LayerConstructor):
                layer = lc.construct(input_shape=input_shape)
            elif isinstance(lc, BaseLayer):
                layer = lc
            elif isinstance(lc, type):
                layer = lc(input_shape=input_shape)
            else:
                raise TypeError(f'Expected LayerConstructor, Layer class or Layer instance, '
                                f'found: {type(lc)}')
            layers.append(layer)
            input_shape = layers[-1].get_output_shape(input_shape)
        model = self.model_class(layers=layers, *self.args, *args, **self.kwargs, **kwargs)

        return model


class LayerConstructor:
    def __init__(self, layer_class, *args, **kwargs):
        self.layer_class = layer_class
        self.args = args
        self.kwargs = kwargs

    def construct(self, *args, **kwargs):
        return self.layer_class(*self.args, *args, **self.kwargs, **kwargs)
