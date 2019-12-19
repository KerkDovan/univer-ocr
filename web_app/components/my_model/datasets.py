import random

import numpy as np
from PIL import Image

from ..nn.gpu import CP
from .constants import (
    INPUT_LAYER_NAME, OUTPUT_LAYER_NAMES, OUTPUT_LAYER_NAMES_PLAIN, OUTPUT_LAYER_NAMES_PLAIN_IDS,
    OUTPUT_LAYER_TAGS, TRAIN_DATA_PATH, TRAIN_DATASET_LENGTH, VALIDATION_DATA_PATH,
    VALIDATION_DATASET_LENGTH)
from .train_data_generator import generate_picture


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


def encode_ys(images):
    ys = []
    idx = 0
    for tag in OUTPUT_LAYER_TAGS:
        y = []
        for _ in OUTPUT_LAYER_NAMES[tag]:
            y.append(np.asarray(images[idx]))
            idx += 1
        y = np.moveaxis(y, 0, -1)
        y = np.reshape(y, (1, *y.shape)) / 255
        ys.append(y)
    return ys


def decode_ys(ys):
    pred_images = []
    thresholded_images = []
    for y in ys:
        y = CP.asnumpy(y)
        y = [y[0, :, :, i] for i in range(y.shape[-1])]
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
        X, ys = encode_X(X_image), encode_ys(y_images)
        return CP.copy(X), [CP.copy(y) for y in ys]

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
            for layer_name in OUTPUT_LAYER_NAMES_PLAIN
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
        X_image = picture[INPUT_LAYER_NAME]
        y_images = [picture[layer_name] for layer_name in OUTPUT_LAYER_NAMES_PLAIN]
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


train_dataset = Dataset(TRAIN_DATASET_LENGTH, TRAIN_DATA_PATH)
validation_dataset = Dataset(VALIDATION_DATASET_LENGTH, VALIDATION_DATA_PATH)


def save_pictures(save_path, X_image, y_images, pred_images, th_images, prefix=''):
    image_monochrome = pred_images[OUTPUT_LAYER_NAMES_PLAIN_IDS['image_monochrome']]
    for i in range(len(pred_images)):
        layer_name = OUTPUT_LAYER_NAMES_PLAIN[i]
        sp = save_path / layer_name
        sp.mkdir(parents=True, exist_ok=True)
        if layer_name == 'image_monochrome':
            this_X_image = X_image
        else:
            this_X_image = image_monochrome
        this_X_image.save(sp / f'{prefix}_1_X.png')
        y_images[i].save(sp / f'{prefix}_2_y.png')
        pred_images[i].save(sp / f'{prefix}_3_pred.png')
        th_images[i].save(sp / f'{prefix}_4_thresholded.png')
