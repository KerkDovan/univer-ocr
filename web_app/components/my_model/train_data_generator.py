import os
import random
from multiprocessing import Process, Queue

import numpy as np

from ..image_generator import LayeredImage, random_font, random_text
from .constants import INPUT_LAYER_NAME, OUTPUT_LAYER_NAMES


def generate_picture(width, height):
    bg_color = tuple(random.randint(1, 255) for _ in range(4))
    layers = LayeredImage(width, height, bg_color)
    for i in range(30):
        layers.add_paragraph(random_text(), random_font(16, 16))
    layers = layers.rotate(random.uniform(0, 360))
    layers = layers.make_divisible_by(16, 16)
    return layers.get_raw()


def generate_train_data(width, height):
    picture = generate_picture(width, height)
    X = np.array(picture[INPUT_LAYER_NAME])
    y = np.array([np.array(picture[name]) for name in OUTPUT_LAYER_NAMES])
    y = np.moveaxis(y, 0, -1)
    X = np.reshape(X, (1, *X.shape)) / 255
    y = np.reshape(y, (1, *y.shape)) / 255
    return X, y


def put_train_data(width, height, queue, generator_func):
    while True:
        train_data = generator_func(width, height)
        queue.put(train_data)


class DataGenerator:
    def __init__(self, width, height, queue_size, generator_func=generate_train_data):
        self.width = width
        self.height = height
        self.queue_size = queue_size
        self.generator_func = generator_func
        self.data_queue = Queue(maxsize=self.queue_size)
        self.workers = [
            Process(target=put_train_data, daemon=True,
                    args=(self.width, self.height, self.data_queue, self.generator_func))
            for _ in range(min(self.queue_size, os.cpu_count()))
        ]

    def start(self):
        for worker in self.workers:
            worker.start()

    def get_data(self):
        result = None
        while result is None:
            try:
                result = self.data_queue.get(timeout=1)
            except Exception:
                pass
        return result
