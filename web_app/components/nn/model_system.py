class BaseComponent:
    def train(self, context):
        raise NotImplementedError()

    def test(self, context):
        raise NotImplementedError()

    def predict(self, context):
        raise NotImplementedError()


class FunctionComponent(BaseComponent):
    def __init__(self, name, func, *args_selectors, **kwargs_selectors):
        self.name = name
        self.func = func
        self.args_selectors = args_selectors
        self.kwargs_selectors = kwargs_selectors

    def __call__(self, context):
        args = [context[v] for v in self.args_selectors]
        kwargs = {k: context[v] for k, v in self.kwargs_selectors.items()}
        context[self.name] = self.func(*args, **kwargs)

    def train(self, context):
        self(context)

    def test(self, context):
        self(context)

    def predict(self, context):
        self(context)


class ModelComponent(BaseComponent):
    def __init__(self, name, model, X_selector, y_selector, pred_selector):
        self.name = name
        self.model = model
        self.X_selector = X_selector
        self.y_selector = y_selector
        self.pred_selector = pred_selector

    def train(self, context):
        context['losses'][self.name] = self.model.train(
            context[self.X_selector], context[self.y_selector])
        context[self.pred_selector] = [
            self.model.layers_outputs[k]
            for k in range(self.model.outputs_count)]

    def test(self, context):
        context['losses'][self.name] = self.model.test(
            context[self.X_selector], context[self.y_selector])
        context[self.pred_selector] = [
            self.model.layers_outputs[k]
            for k in range(self.model.outputs_count)]

    def predict(self, context):
        context['prediction'][self.name] = self.model.predict(
            context[self.X_selector])
        context[self.pred_selector] = [
            self.model.layers_outputs[k]
            for k in range(self.model.outputs_count)]


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
