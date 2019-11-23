from .help_func import make_list_if_not
from .layers import BaseLayer
from .losses import SoftmaxCrossEntropy
from .progress_tracker import track_this


class BaseModel(BaseLayer):
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
    def __init__(self, layers, relations, loss=SoftmaxCrossEntropy(), *args, **kwargs):
        super().__init__(*args, **kwargs)

        if not isinstance(layers, dict):
            raise TypeError(f'layers argument must be dict, found: {type(layers).__name__}')
        if not isinstance(relations, dict):
            raise TypeError(f'relations argument must be dict, found: {type(relations).__name__}')

        self.layers = layers
        self.relations = relations
        self.relations_backward = {}
        self.inputs_count = None
        self.outputs_count = None
        self.loss = loss
        self.input_grads = {}
        self.is_initialized = False

    def initialize(self, input_shapes):
        input_shapes = make_list_if_not(input_shapes)

        self.inputs_count = max(v for k, v in self.relations.items() if isinstance(v, int)) + 1
        self.outputs_count = max(k for k, v in self.relations.items() if isinstance(k, int)) + 1

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
            self.relations[layer_name] = make_list_if_not(self.relations[layer_name])

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
            layer_shapes[layer_name] = self.layers[layer_name].get_output_shapes(
                layer_input_shapes)

            currently_being_visited[layer_name] = False
            return layer_shapes[layer_name]

        for output in output_keys:
            rec_forward_initialize(output)

        never_visited = [name for name, flag in visited.items() if not flag]
        if never_visited:
            print(f'These layers have never been visited: {never_visited}')

        self.is_initialized = True

    @track_this('forward')
    def forward(self, inputs):
        inputs = make_list_if_not(inputs)
        if not self.is_initialized:
            self.initialize_from_X(inputs)

        keys = list(set(self.layers.keys()) | set(self.relations.keys()))
        output_keys = [k for k in keys if isinstance(k, int)]
        outputs = {name: None for name in keys}

        def rec_forward(layer_name):
            if outputs[layer_name] is not None:
                return outputs[layer_name]

            next_inputs = []
            for src in self.relations[layer_name]:
                if isinstance(src, int):
                    next_inputs.append(inputs[src])
                else:
                    next_inputs.append(rec_forward(src))

            if isinstance(layer_name, int):
                outputs[layer_name] = next_inputs[0]
                return

            self.layers[layer_name].clear_grads()
            outputs[layer_name] = self.layers[layer_name].forward(next_inputs)
            if isinstance(outputs[layer_name], list):
                outputs[layer_name] = outputs[layer_name][0]
            return outputs[layer_name]

        for key in output_keys:
            rec_forward(key)

        return [outputs[k] for k in range(self.outputs_count)]

    @track_this('backward')
    def backward(self, grads):
        grads = make_list_if_not(grads)
        keys_backward = list(self.relations_backward.keys())
        grads_mem = {name: None for name in keys_backward}

        def rec_backward(layer_name):
            if grads_mem[layer_name] is not None:
                return grads_mem[layer_name]

            input_grads = []
            for dst, i in self.relations_backward[layer_name].items():
                if isinstance(dst, int):
                    input_grads.append(grads[dst])
                else:
                    input_grads.append(rec_backward(dst)[i])

            input_grads = sum(input_grads)
            if isinstance(layer_name, int):
                grads_mem[layer_name] = input_grads
                return input_grads

            grads_mem[layer_name] = self.layers[layer_name].backward(input_grads)
            grads_mem[layer_name] = make_list_if_not(grads_mem[layer_name])
            return grads_mem[layer_name]

        for key in range(self.inputs_count):
            self.input_grads[key] = rec_backward(key)

        return [self.input_grads[k] for k in range(self.inputs_count)]

    def compute_loss_and_gradients(self, X, y):
        X = make_list_if_not(X)
        y = make_list_if_not(y)

        predicted = self.forward(X)

        losses, gradients = [], []
        for key in range(self.outputs_count):
            loss_func = self.loss[key] if isinstance(self.loss, list) else self.loss
            loss, grad = loss_func(predicted[key], y[key])
            losses.append(loss)
            gradients.append(grad)

        self.backward(gradients)
        reg_loss = self.regularize()

        return {'output_losses': losses, 'regularization_loss': reg_loss}

    def predict(self, X):
        return self.forward(X)

    def update_grads(self):
        for layer in self.layers.values():
            if layer.trainable:
                layer.update_grads()

    def clear_grads(self):
        for layer in self.layers.values():
            layer.clear_grads()

    def get_all_output_shapes(self, input_shapes):
        input_shapes = make_list_if_not(input_shapes)
        output_shapes = {}
        all_output_shapes = {}

        def rec_get_output_shapes(layer_name):
            if layer_name in output_shapes.keys():
                return output_shapes[layer_name]

            layer_input_shapes = []

            for i, src in enumerate(self.relations[layer_name]):
                if isinstance(src, int):
                    layer_input_shapes.append(input_shapes[src])
                else:
                    tmp = rec_get_output_shapes(src)
                    if isinstance(tmp, list):
                        tmp = tmp[0]
                    layer_input_shapes.append(tmp)

            if isinstance(layer_name, int):
                return layer_input_shapes[0]

            tmp = self.layers[layer_name].get_all_output_shapes(layer_input_shapes)
            output_shapes[layer_name] = tmp[0]
            all_output_shapes.update({
                f'{layer_name}/{k}': v for k, v in tmp[1].items()
            })
            return output_shapes[layer_name]

        result = []
        for output in range(self.outputs_count):
            result.append(rec_get_output_shapes(output))
        all_output_shapes.update(output_shapes)
        return result, all_output_shapes

    def get_output_shapes(self, input_shapes):
        return self.get_all_output_shapes(input_shapes)[0]

    def params(self):
        result = {
            f'{layer_name}/{name}': param
            for layer_name, layer in self.layers.items()
            for name, param in layer.params().items()
        }
        return result

    def count_parameters(self):
        return sum([layer.count_parameters() for layer in self.layers.values()])

    def regularize(self):
        total_loss = 0
        for layer in self.layers.values():
            total_loss += layer.regularize()
        return total_loss

    def _set_name(self, name):
        self.name = name
        for layer_name, layer in self.layers.items():
            layer._set_name(f'{self.name}/{layer_name}')

    def init_progress_tracker(self, progress_tracker, set_names_recursively=True):
        if set_names_recursively:
            self._set_name(self.name or 'model')
        else:
            self.name = self.name or 'model'
        self.progress_tracker = progress_tracker
        self.progress_tracker.register_layer(self.name)
        for layer in self.layers.values():
            layer.init_progress_tracker(progress_tracker, set_names_recursively=False)


class Sequential(Model):
    def __init__(self, layers, *args, **kwargs):
        if not isinstance(layers, list):
            raise TypeError(f'layers argument must be list, found: {type(layers).__name__}')

        layers_dict = {}
        relations = {}
        prev_name = 0
        for i, layer in enumerate(layers):
            name = f'{i}_{type(layer).__name__}'
            layers_dict[name] = layer
            relations[name] = prev_name
            prev_name = name
        relations[0] = prev_name

        super().__init__(layers=layers_dict, relations=relations, *args, **kwargs)
