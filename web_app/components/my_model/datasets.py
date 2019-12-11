import random
from pathlib import Path

import numpy as np
from PIL import Image

from ..image_generator import LayeredImage
from ..nn.gpu import CP
from .train_data import generate_picture

DIR_PATH = Path('web_app', 'components', 'my_model')
LAYER_NAMES = [name for name in LayeredImage.layer_names if name != 'image']


def encode_X(image):
    X = np.asarray(image)
    X = np.reshape(X, (1, *X.shape)) / 255
    return X


def decode_X(X):
    if isinstance(X, list):
        X = X[0]
    X = CP.asnumpy(X[0] * 255).astype(np.uint8)
    image = Image.fromarray(X)
    return image


def encode_y(images):
    if not isinstance(images, list):
        images = [images]
    y = []
    for image in images:
        y.append(np.asarray(image))
    y = np.moveaxis(y, 0, -1)
    y = np.reshape(y, (1, *y.shape)) / 255
    return y


def decode_y(y):
    if isinstance(y, list):
        y = y[0]
    y = CP.asnumpy(y)
    y = [y[0, :, :, i] for i in range(y.shape[-1])]
    pred_images = []
    thresholded_images = []
    for yi in y:
        cm = np.mean(yi)
        thresholded = ((yi >= cm) * 255).astype(np.uint8)
        yi = (yi * 255).astype(np.uint8)
        pred_image = Image.fromarray(yi)
        thresholded_image = Image.fromarray(thresholded)
        pred_images.append(pred_image)
        thresholded_images.append(thresholded_image)
    return pred_images, thresholded_images


class BaseDataset:
    def __init__(self, size):
        self.size = size

    def get(self, idx, X_image=None, y_images=None):
        if X_image is None or y_images is None:
            X_image, y_images = self.get_images(idx)
        X, y = encode_X(X_image), encode_y(y_images)
        return CP.copy(X), CP.copy(y)

    def get_images(self, idx):
        raise NotImplementedError()

    def __len__(self):
        return self.size


class Dataset(BaseDataset):
    def __init__(self, size, dirpath):
        super().__init__(size)
        self.dirpath = dirpath

    def get_images(self, idx):
        X_path = self.dirpath / f'{idx}_image.png'
        y_paths = [
            self.dirpath / f'{idx}_{layer_name}.png'
            for layer_name in LAYER_NAMES
        ]
        X_image = Image.open(X_path)
        y_images = [Image.open(y_path) for y_path in y_paths]
        return X_image, y_images


class GeneratorDataset(BaseDataset):
    def __init__(self, size, width, height):
        super().__init__(size)
        self.width = width
        self.height = height

    def get_images(self, idx, width=None, height=None):
        width = self.width if width is None else width
        height = self.height if height is None else height
        picture = generate_picture(width, height)
        X_image = picture['image']
        y_images = [picture[layer_name] for layer_name in LAYER_NAMES]
        return X_image, y_images


class RandomSelectDataset(BaseDataset):
    def __init__(self, size, source_dataset):
        self.size = size
        self.source_dataset = source_dataset
        self.selected = []
        while len(self.selected) < self.size:
            idx = random.choice(range(len(source_dataset)))
            if idx not in self.selected:
                self.selected.append(idx)

    def get_images(self, idx):
        return self.source_dataset.get_images(self.selected[idx])


train_dataset = Dataset(10000, DIR_PATH / 'data' / 'train')
validation_dataset = Dataset(1000, DIR_PATH / 'data' / 'validation')


def save_pictures(save_path, X_image, y_images, pred_images, th_images, prefix=''):
    for i, layer_name in enumerate(LAYER_NAMES):
        sp = save_path / layer_name
        sp.mkdir(parents=True, exist_ok=True)
        X_image.save(sp / f'{prefix}_1_X.png')
        y_images[i].save(sp / f'{prefix}_2_y.png')
        pred_images[i].save(sp / f'{prefix}_3_pred.png')
        th_images[i].save(sp / f'{prefix}_4_thresholded.png')