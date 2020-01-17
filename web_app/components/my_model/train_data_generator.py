import os
import random
from multiprocessing import Event, Process, Queue
from queue import Empty, Full

import numpy as np

from ..image_generator import LayeredImage, random_font, random_text
from .constants import LAYER_TAGS, LAYER_NAMES


def generate_picture(width, height, rotate=False):
    bg_color = tuple(random.randint(1, 255) for _ in range(4))
    layers = LayeredImage(width, height, bg_color)
    for i in range(30):
        layers.add_paragraph(random_text(), random_font(16, 16))
    if rotate:
        layers = layers.rotate(random.uniform(0, 360))
    layers = layers.make_divisible_by(16, 16)
    return layers.get_raw()


def encode_layers(images):
    layers = {}
    for tag in LAYER_TAGS:
        layer = np.array([
            np.asarray(images[layer_name].convert('L'))
            for layer_name in LAYER_NAMES[tag]
        ])
        layer = np.moveaxis(layer, 0, -1)
        layer = np.reshape(layer, (1, *layer.shape)) / 255
        layers[tag] = layer
    return layers


def generate_train_data(width, height, rotate=False):
    return encode_layers(generate_picture(width, height, rotate))


class DataGenerator:
    def __init__(self, queue_size=None, generator_func=generate_train_data,
                 func_args=(), func_kwargs={}):
        self.queue_size = os.cpu_count() if queue_size is None else queue_size
        self.generator_func = generator_func
        self.func_args = func_args
        self.func_kwargs = func_kwargs
        self.done = Event()
        self.data_queue = Queue(maxsize=self.queue_size)
        self.workers = [
            Process(target=self._run, daemon=True, args=(
                self.done, self.data_queue,
                self.generator_func, self.func_args, self.func_kwargs))
            for _ in range(min(self.queue_size, os.cpu_count()))
        ]

    def start(self):
        self.done.clear()
        for worker in self.workers:
            worker.start()

    def stop(self):
        self.done.set()

    @staticmethod
    def _run(done, queue, generator_func, func_args, func_kwargs):
        train_data = None
        while not done.is_set():
            if train_data is None:
                train_data = generator_func(*func_args, **func_kwargs)
            try:
                queue.put(train_data, timeout=0.1)
                train_data = None
            except Full:
                pass

    def get_data(self):
        result = None
        while result is None and not self.done.is_set():
            try:
                result = self.data_queue.get(timeout=0.1)
            except Empty:
                pass
        return result
