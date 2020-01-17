import os
import random
import shutil

import numpy as np
from PIL import Image

from ..nn.gpu import CP
from ..nn.help_func import make_list_if_not
from .constants import (
    LAYER_NAMES, LAYER_NAMES_PLAIN, LAYER_NAMES_PLAIN_IDS, LAYER_TAGS, TRAIN_DATA_PATH,
    TRAIN_DATASET_LENGTH, VALIDATION_DATA_PATH, VALIDATION_DATASET_LENGTH)
from .train_data_generator import generate_picture


def encode_X(image):
    X = np.asarray(image)
    X = np.reshape(X, (1, *X.shape, 1)) / 255
    return X


def decode_X(X):
    if isinstance(X, list):
        X = X[0]
    X = CP.asnumpy(X[0, :, :, 0] * 255).astype(np.uint8)
    image = Image.fromarray(X)
    return image


def encode_ys(images):
    ys = []
    idx = 0
    for tag in LAYER_TAGS:
        y = []
        for _ in LAYER_NAMES[tag]:
            y.append(np.asarray(images[idx]))
            idx += 1
        y = np.moveaxis(y, 0, -1)
        y = np.reshape(y, (1, *y.shape)) / 255
        ys.append(y)
    return ys


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


def decode_y(y, normalize=False):
    pred_images = []
    thresholded_images = []
    y = CP.asnumpy(y)
    y = [y[0, :, :, i] for i in range(y.shape[-1])]
    for yi in y:
        if normalize:
            yi -= np.min(yi)
            max_val = np.max(yi)
            if not np.isclose(max_val, 0):
                yi /= max_val
        cm = np.mean(yi)
        thresholded = ((yi >= cm) * 255).astype(np.uint8)
        yi = (yi * 255).astype(np.uint8)
        pred_image = Image.fromarray(yi)
        thresholded_image = Image.fromarray(thresholded)
        pred_images.append(pred_image)
        thresholded_images.append(thresholded_image)
    return pred_images, thresholded_images


def decode_ys(ys, normalize=False):
    pred_images = []
    thresholded_images = []
    for y in ys:
        p, th = decode_y(y, normalize)
        pred_images.extend(p)
        thresholded_images.extend(th)
    return pred_images, thresholded_images


def get_layer_names(layer_tags=None):
    layer_names = [
        name
        for tag in LAYER_TAGS
        if layer_tags is None or tag in layer_tags
        for name in LAYER_NAMES[tag]
    ]
    return layer_names


class BaseDataset:
    def __init__(self, size):
        self.size = size

    def get(self, idx, layer_images=None, layer_tags=None):
        if layer_images is None:
            layer_images = self.get_images(idx, layer_tags=layer_tags)
        elif layer_tags is not None:
            layer_names = get_layer_names(layer_tags)
            layer_images = {name: layer_images[name] for name in layer_names}
        layers = encode_layers(layer_images)
        return layers

    def get_images(self, idx, layer_tags=None):
        raise NotImplementedError()

    def __len__(self):
        return self.size


class Dataset(BaseDataset):
    def __init__(self, size, dirpath):
        super().__init__(size)
        self.dirpath = dirpath

    def get_images(self, idx, layer_tags=None):
        layer_names = get_layer_names(layer_tags)
        layer_paths = {
            layer_name: self.dirpath / f'{idx}_{layer_name}.png'
            for layer_name in LAYER_NAMES_PLAIN
            if layer_tags is None or layer_name in layer_names
        }
        layer_images = {
            layer_name: Image.open(layer_path).convet('L')
            for layer_name, layer_path in layer_paths.items()
        }
        return layer_images


class GeneratorDataset(BaseDataset):
    def __init__(self, size, width, height):
        super().__init__(size)
        self.width = width
        self.height = height

    def get_images(self, idx, layer_tags=None, width=None, height=None, rotate=False):
        width = self.width if width is None else width
        height = self.height if height is None else height
        picture = generate_picture(width, height, rotate)
        layer_names = get_layer_names(layer_tags)
        layer_images = {
            layer_name: image.convert('L')
            for layer_name, image in picture.items()
            if layer_name in layer_names
        }
        return layer_images


class RandomSelectDataset(BaseDataset):
    def __init__(self, size, source_dataset):
        self.size = size
        self.source_dataset = source_dataset
        self.selected = []
        while len(self.selected) < self.size:
            idx = random.choice(range(len(source_dataset)))
            if idx not in self.selected:
                self.selected.append(idx)

    def get_images(self, idx, layer_tags=None):
        return self.source_dataset.get_images(self.selected[idx], layer_tags=layer_tags)


train_dataset = Dataset(TRAIN_DATASET_LENGTH, TRAIN_DATA_PATH)
validation_dataset = Dataset(VALIDATION_DATASET_LENGTH, VALIDATION_DATA_PATH)


def save_pictures(save_path, X_image, y_images, pred_images, th_images, prefix=''):
    return
    image_monochrome = pred_images[LAYER_NAMES_PLAIN_IDS['image_monochrome']]
    for i in range(len(pred_images)):
        layer_name = LAYER_NAMES_PLAIN[i]
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


def save_layers_outputs_pictures(save_path, layers_outputs):
    return
    if save_path.exists():
        for path in save_path.iterdir():
            if path.is_dir():
                shutil.rmtree(path)
            else:
                os.remove(path)
    for layer_name, outputs in layers_outputs.items():
        if isinstance(layer_name, int):
            continue
        layer_save_path = save_path / layer_name.replace('/', '_')
        outputs = make_list_if_not(outputs)
        for output_id in range(len(outputs)):
            pred_save_path = layer_save_path / f'output_{output_id}_predicted'
            th_save_path = layer_save_path / f'output_{output_id}_thresholded'
            pred_save_path.mkdir(parents=True, exist_ok=True)
            th_save_path.mkdir(parents=True, exist_ok=True)
            output = [outputs[output_id]]
            pred_images, th_images = decode_ys(output, normalize=True)
            for channel, (pred, th) in enumerate(zip(pred_images, th_images)):
                pred.save(pred_save_path / f'channel_{channel}.png')
                th.save(th_save_path / f'channel_{channel}.png')
