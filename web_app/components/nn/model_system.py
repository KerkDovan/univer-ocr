class BaseComponent:
    def train(self, context):
        raise NotImplementedError()

    def test(self, context):
        raise NotImplementedError()

    def predict(self, context):
        raise NotImplementedError()


class RawFunctionComponent(BaseComponent):
    def __init__(self, func):
        self.func = func

    def __call__(self, context):
        self.func(context)

    def train(self, context):
        self(context)

    def test(self, context):
        self(context)

    def predict(self, context):
        self(context)


class WrappedFunctionComponent(RawFunctionComponent):
    def __init__(self, name, func, *args_labels, **kwargs_labels):
        super().__init__(func)
        self.name = name
        self.args_labels = args_labels
        self.kwargs_labels = kwargs_labels

    def __call__(self, context):
        args = [context[v] for v in self.args_labels]
        kwargs = {k: context[v] for k, v in self.kwargs_labels.items()}
        context[self.name] = self.func(*args, **kwargs)


class BaseSelector:
    def __init__(self):
        self.context = None

    def __call__(self, context):
        self.context = context

    def get(self):
        raise NotImplementedError()

    def get_X(self):
        raise NotImplementedError()

    def put(self, pred):
        raise NotImplementedError()


class StringSelector(BaseSelector):
    def __init__(self, X_label, y_label, pred_label):
        super().__init__()
        self.X_label = X_label
        self.y_label = y_label
        self.pred_label = pred_label

    def get(self):
        yield self.context[self.X_label], self.context[self.y_label]

    def get_X(self):
        yield self.context[self.X_label]

    def put(self, pred):
        self.context[self.pred_label] = pred


class IterableSelector(BaseSelector):
    def __init__(self, X_label, y_label, pred_label):
        super().__init__()
        self.X_label = X_label
        self.y_label = y_label
        self.pred_label = pred_label

    def get(self):
        for X, y in zip(self.context[self.X_label], self.context[self.y_label]):
            yield X, y

    def get_X(self):
        for X in self.context[self.X_label]:
            yield X

    def put(self, pred):
        if self.pred_label not in self.context.keys():
            self.context[self.pred_label] = []
        self.context[self.pred_label].append(pred)


class ModelComponent(BaseComponent):
    def __init__(self, name, model, selector, delist_result=False):
        self.name = name
        self.model = model
        self.selector = selector
        self.delist_result = delist_result

    def train(self, context):
        self.selector(context)
        for X, y in self.selector.get():
            losses = self.model.train(X, y)
            if self.name not in context['losses']:
                context['losses'][self.name] = losses
            else:
                for k, v in losses.items():
                    context['losses'][self.name][k] += v
            result = [
                self.model.layers_outputs[k]
                for k in range(self.model.outputs_count)]
            if self.delist_result:
                result = result[0]
            self.selector.put(result)

    def test(self, context):
        self.selector(context)
        for X, y in self.selector.get():
            losses = self.model.test(X, y)
            if self.name not in context['losses']:
                context['losses'][self.name] = losses
            else:
                for k, v in losses.items():
                    context['losses'][self.name][k] += v
            result = [
                self.model.layers_outputs[k]
                for k in range(self.model.outputs_count)]
            if self.delist_result:
                result = result[0]
            self.selector.put(result)

    def predict(self, context):
        self.selector(context)
        for X in self.selector.get_X():
            context['prediction'][self.name] = self.model.predict(X)
            result = [
                self.model.layers_outputs[k]
                for k in range(self.model.outputs_count)]
            if self.delist_result:
                result = result[0]
            self.selector.put(result)


class ModelSystem:
    def __init__(self, components):
        assert isinstance(components, list)
        assert all(isinstance(c, BaseComponent) for c in components)
        self.components = components

    def train(self, context):
        context['losses'] = {}
        for component in self.components:
            component.train(context)

    def test(self, context):
        context['losses'] = {}
        for component in self.components:
            component.test(context)

    def predict(self, context):
        context['prediction'] = {}
        for component in self.components:
            component.predict(context)
