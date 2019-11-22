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


class Model(BaseModel):
    def __init__(self, layers, relations, optimizer=Adam(), loss=SoftmaxCrossEntropy()):
        if not isinstance(layers, dict):
            raise TypeError(f'layers argument must be dict, found: {type(layers)}')
        if not isinstance(relations, dict):
            raise TypeError(f'relations argument must be dict, found: {type(relations)}')
        self.layers = layers
        self.relations = relations
        self.relations_backward = {}
        self.optimizer = optimizer
        self.loss = loss
        self.input_grads = {}
        self.is_initialized = False

    def initialize_from_X(self, X):
        if not isinstance(X, list):
            X = [X]
        self.initialize([x.shape for x in X])

    def initialize(self, input_shapes):
        if not isinstance(input_shapes, list):
            input_shapes = [input_shapes]

        keys = list(set(self.layers.keys()) | set(self.relations.keys()))
        output_keys = [k for k in keys if isinstance(k, int)]

        visited = {name: False for name in keys}
        currently_being_visited = {name: False for name in keys}
        layer_shapes = {name: None for name in keys}

        def rec_forward_initialize(layer_name):
            visited[layer_name] = True
            if currently_being_visited[layer_name]:
                raise RecursionError(f'Looped on {layer_name} layer, check relations')
            if layer_shapes[layer_name] is not None:
                return layer_shapes[layer_name]
            currently_being_visited[layer_name] = True

            layer_input_shapes = []
            if not isinstance(self.relations[layer_name], list):
                self.relations[layer_name] = [self.relations[layer_name]]

            for i, src in enumerate(self.relations[layer_name]):
                if isinstance(src, int):
                    layer_input_shapes.append(input_shapes[src])
                else:
                    tmp = rec_forward_initialize(src)
                    if isinstance(tmp, list):
                        tmp = tmp[0]
                    layer_input_shapes.append(tmp)

                if src not in self.relations_backward:
                    self.relations_backward[src] = {}
                self.relations_backward[src][layer_name] = i

            if isinstance(layer_name, int):
                return

            if not self.layers[layer_name].is_initialized:
                self.layers[layer_name].initialize(layer_input_shapes)
            layer_shapes[layer_name] = self.layers[layer_name].get_output_shape(layer_input_shapes)

            currently_being_visited[layer_name] = False
            return layer_shapes[layer_name]

        for output in output_keys:
            rec_forward_initialize(output)

        never_visited = [name for name, flag in visited.items() if not flag]
        if never_visited:
            print(f'These layers have never been visited: {never_visited}')

        self.is_initialized = True

    def compute_loss_and_gradients(self, X, y):
        if not isinstance(X, list):
            X = [X]
        if not isinstance(y, list):
            y = [y]

        keys = list(set(self.layers.keys()) | set(self.relations.keys()))
        output_keys = sorted([k for k in keys if isinstance(k, int)])
        predicted = self.predict(X)

        losses, gradients = [], []
        for key in output_keys:
            loss_func = self.loss[key] if isinstance(self.loss, list) else self.loss
            loss, grad = loss_func(predicted[key], y[key])
            losses.append(loss)
            gradients.append([grad])

        keys_backward = list(self.relations_backward.keys())
        input_keys = sorted([k for k in keys_backward if isinstance(k, int)])

        grads = {name: None for name in keys_backward}

        def rec_backward(layer_name):
            if grads[layer_name] is not None:
                return grads[layer_name]

            input_grads = []
            for dst, i in self.relations_backward[layer_name].items():
                if isinstance(dst, int):
                    input_grads.append(gradients[dst][0])
                else:
                    input_grads.append(rec_backward(dst)[i])

            input_grads = sum(input_grads)
            if isinstance(layer_name, int):
                grads[layer_name] = input_grads
                return input_grads

            grads[layer_name] = self.layers[layer_name].backward(input_grads)
            if not isinstance(grads[layer_name], list):
                grads[layer_name] = [grads[layer_name]]
            return grads[layer_name]

        for key in input_keys:
            self.input_grads[key] = rec_backward(key)

        return losses

    def predict(self, X):
        if not isinstance(X, list):
            X = [X]
        if not self.is_initialized:
            self.initialize_from_X(X)

        keys = list(set(self.layers.keys()) | set(self.relations.keys()))
        output_keys = [k for k in keys if isinstance(k, int)]
        outputs = {name: None for name in keys}

        def rec_forward(layer_name):
            if outputs[layer_name] is not None:
                return outputs[layer_name]

            inputs = []
            for src in self.relations[layer_name]:
                if isinstance(src, int):
                    inputs.append(X[src])
                else:
                    inputs.append(rec_forward(src))

            if isinstance(layer_name, int):
                outputs[layer_name] = inputs[0]
                return

            self.layers[layer_name].clear_grads()
            outputs[layer_name] = self.layers[layer_name].forward(inputs)
            if isinstance(outputs[layer_name], list):
                outputs[layer_name] = outputs[layer_name][0]
            return outputs[layer_name]

        for key in output_keys:
            rec_forward(key)

        return outputs

    def params(self):
        result = {
            f'{layer_name}_{name}': param
            for layer_name, layer in self.layers.items()
            for name, param in layer.params().items()
        }
        return result

    def count_parameters(self):
        return sum([layer.count_parameters() for layer in self.layers.values()])


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
