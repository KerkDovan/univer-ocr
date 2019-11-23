from datetime import datetime
from functools import wraps


class Event:
    def __init__(self, name):
        self.name = name
        self.done = False
        self.started = None
        self.stopped = None
        self.time = None

    def start(self):
        self.started = datetime.now()

    def stop(self):
        self.stopped = datetime.now()
        self.time = self.stopped - self.started
        self.done = True

    def to_dict(self):
        return {
            'name': self.name,
            'done': self.done,
            'started': self.started,
            'stopped': self.stopped,
            'time': self.time,
        }


class BaseProgressTracker:
    def __init__(self, *args, **kwargs):
        pass

    def register_layer(self, name):
        pass

    def get_summary(self):
        return {}

    def start_tracking(self, name, event):
        pass

    def stop_tracking(self, name, event):
        pass


class ProgressTracker(BaseProgressTracker):
    def __init__(self, handler=print):
        self.layers = {}
        self.handler = handler

    def register_layer(self, name):
        self.layers[name] = {}

    def get_summary(self):
        return {
            name: [event.to_dict() for event in layer.values()]
            for name, layer in self.layers.items()
        }

    def start_tracking(self, name, event):
        self.layers[name][event] = Event(event)
        self.layers[name][event].start()
        self.handler(self.get_summary())

    def stop_tracking(self, name, event):
        self.layers[name][event].stop()
        self.handler(self.get_summary())


def track_this(event):
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            self.progress_tracker.start_tracking(self.name, event)
            result = func(self, *args, **kwargs)
            self.progress_tracker.stop_tracking(self.name, event)
            return result
        return wrapper
    return decorator
