from .losses import SoftmaxCrossEntropy
from .optimizers import Adam


class Model:
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


class Sequential(Model):
    def __init__(self, layers, optimizer=Adam(), loss=SoftmaxCrossEntropy()):
        self.layers = layers
        self.optimizer = optimizer
        self.loss = loss

    def compute_loss_and_gradients(self, X, y):
        pred = X
        for layer in self.layers:
            layer.clear_grads()
            pred = layer.forward(pred)

        loss, grad = self.loss(pred, y)

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
