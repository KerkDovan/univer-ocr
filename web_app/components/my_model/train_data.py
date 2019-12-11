from multiprocessing import Process, Queue
import os

import numpy as np

from ..image_generator import LayeredImage, random_font, random_text


def generate_picture(width, height):
    bg_color = (255, 255, 255, 255)
    layers = LayeredImage(width, height, bg_color)
    for i in range(30):
        layers.add_paragraph(random_text(), random_font(16, 16))
    return layers.get_raw()


def generate_train_data(width, height):
    picture = generate_picture(width, height)
    layer_names = ['char_mask_box']  # LayeredImage.layer_names
    X = np.array(picture['image'])
    y = np.array([np.array(picture[name])
                  for name in layer_names
                  if name != 'image'])
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
