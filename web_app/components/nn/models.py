from .help_func import make_list_if_not
from .layers import BaseLayer
from .losses import SoftmaxCrossEntropy
from .progress_tracker import track_method


class BaseModel(BaseLayer):
    def compute_loss_and_gradients(self, X, y):
        raise NotImplementedError()

    def train(self, X, y):
        loss = self.compute_loss_and_gradients(X, y)
        for param in self.params().values():
            param.update_grad()
            param.clear_grad()
        return loss

    def test(self, X, y):
        raise NotImplementedError()

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

        self.ravelled_layers = layers
        self.ravelled_relations = relations
        self.layers = None
        self.relations = None
        self.relations_backward = {}
        self.inputs_count = max(v for k, v in relations.items() if isinstance(v, int)) + 1
        self.outputs_count = max(k for k, v in relations.items() if isinstance(k, int)) + 1
        self.layers_outputs = {}
        self.loss = loss
        self.input_grads = {}
        self.is_initialized = False
        self._receptive_fields = {}

        self.unravel_model()

    def initialize(self, input_shapes):
        input_shapes = make_list_if_not(input_shapes)
        self.input_shapes = input_shapes

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

    def unravel_model(self):
        relations = {dst: make_list_if_not(src) for dst, src in self.ravelled_relations.items()}
        for layer_name, layer in self.ravelled_layers.items():
            if not isinstance(layer, Model):
                continue

            layer.unravel_model()

            # sources of layer - destinations of self
            layer_relations = {dst: srcs for dst, srcs in layer.relations.items()}
            new_layer_relations = {}
            for dst, srcs in layer_relations.items():
                new_srcs = []
                for src in srcs:
                    if isinstance(src, int):
                        new_srcs.append(relations[layer_name][src])

                    else:
                        new_srcs.append(f'{layer_name}/{src}')
                dst_name = dst if isinstance(dst, int) else f'{layer_name}/{dst}'
                new_layer_relations[dst_name] = new_srcs

            # destinations of layer - sources of self
            for dst, srcs in relations.items():
                new_srcs = []
                for src in srcs:
                    if isinstance(src, str) and layer_name == src:
                        for out_id in range(layer.get_outputs_count()):
                            for unravelled_src in new_layer_relations[out_id]:
                                new_srcs.append(unravelled_src)

                    elif isinstance(src, tuple) and len(src) > 1 and layer_name == src[0]:
                        for out_id in src[1:]:
                            for unravelled_src in new_layer_relations[out_id]:
                                new_srcs.append(unravelled_src)

                    else:
                        new_srcs.append(src)
                relations[dst] = new_srcs

            for out_id in range(layer.get_outputs_count()):
                del new_layer_relations[out_id]
            relations.update(new_layer_relations)
            del relations[layer_name]

        self.layers = self.get_leaf_layers()
        self.relations = relations

        for layer_name, layer in self.layers.items():
            layer._set_name(layer_name)

    def __getitem__(self, key):
        return self.layers[key]

    @track_method('forward')
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

        self.layers_outputs = outputs

        return [outputs[k] for k in range(self.outputs_count)]

    @track_method('backward')
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

    def train(self, X, y):
        losses = self.compute_loss_and_gradients(X, y)
        self.update_grads()
        self.clear_grads()
        return losses

    def test(self, X, y):
        X = make_list_if_not(X)
        y = make_list_if_not(y)

        predicted = self.forward(X)

        losses = []
        for key in range(self.outputs_count):
            loss_func = self.loss[key] if isinstance(self.loss, list) else self.loss
            loss, _ = loss_func(predicted[key], y[key])
            losses.append(loss)

        return {'output_losses': losses}

    def predict(self, X):
        return self.forward(X)

    def update_grads(self):
        if not self.trainable:
            return
        for layer in self.layers.values():
            layer.update_grads()

    def clear_grads(self):
        for layer in self.layers.values():
            layer.clear_grads()
        self.input_grads = {}

    def get_all_output_shapes(self, input_shapes):
        input_shapes = make_list_if_not(input_shapes)
        output_shapes = {}
        all_output_shapes = {}

        def from_numpy(shapes):
            shapes = make_list_if_not(shapes)
            result = []
            for shape in shapes:
                assert isinstance(shape, tuple)
                result.append(tuple(int(x) for x in shape))
            return result

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
            output_shapes[layer_name] = from_numpy(tmp[0])
            all_output_shapes.update({
                f'{layer_name}/{k}': from_numpy(v) for k, v in tmp[1].items()
            })
            return output_shapes[layer_name]

        result = []
        for output in range(self.outputs_count):
            result.append(rec_get_output_shapes(output))
        all_output_shapes.update(output_shapes)
        return from_numpy(result), all_output_shapes

    def get_output_shapes(self, input_shapes):
        return self.get_all_output_shapes(input_shapes)[0]

    def get_outputs_count(self):
        return self.outputs_count

    def is_fully_convolutional(self):
        return all(layer.is_fully_convolutional() for layer in self.layers.values())

    def changes_receptive_field(self):
        return any(layer.changes_receptive_field() for layer in self.layers.values())

    def get_receptive_fields(self):
        assert self.is_initialized, (
            f'The model must be initialized before calling this method')
        assert self.is_fully_convolutional(), (
            f'This method is only available for Fully Convolutional Networks (FCN)')

        for output_id in range(self.get_outputs_count()):
            for axis in range(2):
                self._get_receptive_field(axis, 0, output_id)

        tmp = {
            layer_name: (
                self._receptive_fields[layer_name, 0],
                self._receptive_fields[layer_name, 1])
            for layer_name in self._receptive_fields['relations'].keys()
            if not isinstance(layer_name, int)
        }
        result = {}
        for layer_name, (rf_y, rf_x) in tmp.items():
            result[layer_name] = {}
            for in_id in rf_y.keys():
                rf1_y, rf1_x = rf_y[in_id], rf_x[in_id]
                cnt_y, cnt_x = len(rf1_y), len(rf1_x)
                min_y, max_y = min(rf1_y), max(rf1_y)
                min_x, max_x = min(rf1_x), max(rf1_x)
                result[layer_name][f'input {in_id}'] = {
                    'cnt': (cnt_y, cnt_x),
                    'y': (min_y, max_y),
                    'x': (min_x, max_x),
                    'is_solid_y': (cnt_y == max_y - min_y + 1),
                    'is_solid_x': (cnt_x == max_x - min_x + 1),
                }

        self._clear_receptive_fields_info()
        return result

    def _get_receptive_field(self, axis, position, output_id):
        if (axis, position, output_id) in self._receptive_fields:
            return self._receptive_fields[axis, position, output_id]

        if 'relations' in self._receptive_fields:
            relations = self._receptive_fields['relations']
        else:
            relations = {dst: srcs for dst, srcs in self.relations.items()}
            for layer_name, layer in self.layers.items():
                if layer.changes_receptive_field():
                    continue
                sources = relations[layer_name]
                destinations = [dst for dst, src in relations.items()
                                if layer_name == src or layer_name in src]
                for dst in destinations:
                    if relations[dst] == layer_name:
                        relations[dst] = sources
                    else:
                        tmp = []
                        for src in relations[dst]:
                            tmp.extend(sources if src == layer_name else [src])
                        relations[dst] = tmp
                del relations[layer_name]
            self._receptive_fields['relations'] = relations

        input_keys = list(range(self.inputs_count))
        all_input_points = {}

        def rec_get_receptive_field(layer_name, axis, pos, out_id):
            if (layer_name, axis, pos, out_id) in all_input_points:
                return all_input_points[layer_name, axis, pos, out_id]
            if isinstance(layer_name, int):
                points = {0: set([pos])}
            else:
                points = self.layers[layer_name]._get_receptive_field(axis, pos, out_id)
            input_points = {in_key: set() for in_key in input_keys}
            for src_id, src in enumerate(relations[layer_name]):
                if isinstance(src, int):
                    input_points[src].update(points[src_id])
                    continue
                for point in points[src_id]:
                    src_input_points = rec_get_receptive_field(src, axis, point, 0)
                    for in_key, in_points in src_input_points.items():
                        input_points[in_key].update(in_points)
            all_input_points[layer_name, axis, pos, out_id] = input_points
            return all_input_points[layer_name, axis, pos, out_id]

        for layer_name in relations.keys():
            self._receptive_fields[layer_name, axis] = rec_get_receptive_field(
                layer_name, axis, 0, 0)

        return rec_get_receptive_field(relations[output_id][0], axis, position, 0)

    def _clear_receptive_fields_info(self):
        for layer_name, layer in self.layers.items():
            layer._clear_receptive_fields_info()
        self._receptive_fields = {}

    def get_leaf_layers(self):
        if self.layers is not None:
            return self.layers
        result = {}
        for layer_name, layer in self.ravelled_layers.items():
            if isinstance(layer, Model):
                submodel_layers = layer.get_leaf_layers()
                for name, sub_layer in submodel_layers.items():
                    result[f'{layer_name}/{name}'] = sub_layer
            else:
                result[layer_name] = layer
        return result

    def params(self):
        result = {
            f'{layer_name}/{name}': param
            for layer_name, layer in self.layers.items()
            for name, param in layer.params().items()
        }
        return result

    def get_weights(self):
        all_weights = {name: layer.get_weights() for name, layer in self.layers.items()}
        return {name: weights for name, weights in all_weights.items() if weights != {}}

    def set_weights(self, weights):
        for name, layer in self.layers.items():
            layer_weights = weights.get(name, None)
            if layer_weights is None:
                continue
            layer.set_weights(layer_weights)

    def nan_weights(self):
        return any(layer.nan_weights() for layer in self.layers.values())

    def count_parameters(self):
        return sum([layer.count_parameters() for layer in self.layers.values()])

    def regularize(self):
        total_loss = 0
        for layer in self.layers.values():
            total_loss += layer.regularize()
        return total_loss

    def init_progress_tracker(self, progress_tracker, model_name='model'):
        if self.name is None:
            self.name = model_name
        self.progress_tracker = progress_tracker
        self.progress_tracker.register_layer(self.name)
        for layer in self.layers.values():
            layer.init_progress_tracker(progress_tracker, None)


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
