import json
import random

from ..nn.gpu import CP
from .constants import LAYERS_OUTPUTS_PATH, MODEL_WEIGHTS_FILE_PATH, PREDICTION_RESULT_PATH
from .datasets import (
    GeneratorDataset, decode_ys, save_layers_outputs_pictures, save_pictures, validation_dataset)
from .model import make_model


def load_model(input_shape):
    model_weights_file = MODEL_WEIGHTS_FILE_PATH
    try:
        weights = json.load(open(model_weights_file, 'r'))
    except OSError:
        print('No model_weights.json file found')
        weights = {}

    model = make_model(input_shape)
    model.initialize(input_shape)
    model.set_weights(weights)
    return model


def main(use_gpu=False, generate=False, save_layers_outputs=False):
    if use_gpu:
        CP.use_gpu()
        print('Using GPU')
    else:
        CP.use_cpu()
        print('Using CPU')

    if generate:
        dataset = GeneratorDataset(1000, 640, 480)
        print('Using generated data')
    else:
        dataset = validation_dataset
        print('Using validation dataset')
    idx = random.randint(0, len(dataset) - 1)
    print(f'Data #{idx}')

    X_image, y_images = dataset.get_images(idx)
    X, y = dataset.get(idx, X_image, y_images)

    input_shape = X.shape
    print(f'Input shape: {input_shape}')

    model = load_model(input_shape)

    prediction = model.predict(X)
    y_images, _ = decode_ys(y)
    pred_images, th_images = decode_ys(prediction)

    save_path = PREDICTION_RESULT_PATH
    save_pictures(save_path, X_image, y_images, pred_images, th_images)

    if save_layers_outputs is True:
        save_layers_outputs_pictures(LAYERS_OUTPUTS_PATH, model.layers_outputs)
