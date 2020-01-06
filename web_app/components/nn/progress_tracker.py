from datetime import datetime
from functools import wraps


class Event:
    def __init__(self, name):
        self.name = name
        self.done = False
        self.started = None
        self.stopped = None
        self.time = None
        self.counter = 0

    def start(self):
        self.done = False
        self.started = datetime.now()

    def stop(self):
        self.stopped = datetime.now()
        time = self.stopped - self.started
        self.time = time if self.time is None else self.time + time
        self.done = True
        self.counter += 1

    def reset(self):
        self.done = False
        self.started = None
        self.stopped = None
        self.time = None
        self.counter = 0

    def to_dict(self):
        return {
            'name': self.name,
            'done': self.done,
            'started': self.started,
            'stopped': self.stopped,
            'time': self.time,
            'counter': self.counter,
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

    def message(self, message):
        pass

    def reset(self):
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
        if event not in self.layers[name]:
            self.layers[name][event] = Event(event)
        self.layers[name][event].start()
        self.handler(event, self.get_summary())

    def stop_tracking(self, name, event):
        self.layers[name][event].stop()
        self.handler(event, self.get_summary())

    def message(self, message, data=None):
        self.handler(message, data)

    def reset(self):
        self.handler('reset')
        for events in self.layers.values():
            for event in events.values():
                event.reset()


def track_method(event):
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            self.progress_tracker.start_tracking(self.name, event)
            result = func(self, *args, **kwargs)
            self.progress_tracker.stop_tracking(self.name, event)
            return result
        return wrapper
    return decorator


def track_function(name, event, progress_tracker):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            progress_tracker.start_tracking(name, event)
            result = func(*args, **kwargs)
            progress_tracker.stop_tracking(name, event)
            return result
        return wrapper
    progress_tracker.register_layer(name)
    return decorator
