import json
import random

from ..nn.gpu import CP
from .datasets import DIR_PATH, GeneratorDataset, decode_y, save_pictures, validation_dataset
from .model import make_unet


def load_model(input_shape, output_shape):
    model_weights_file = DIR_PATH / 'model_weights.json'
    try:
        weights = json.load(open(model_weights_file, 'r'))
    except OSError:
        print('No model_weights.json file found')
        weights = {}

    model = make_unet(input_shape[3], output_shape[3])
    model.initialize(input_shape)
    model.set_weights(weights)
    return model


def predict(X, model):
    pred = model.predict(X)[0]
    pred_images, th_images = decode_y(pred)
    return pred_images, th_images


def main(use_gpu=False, generate=False):
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

    input_shape, output_shape = X.shape, y.shape
    print(f'Input shape: {input_shape}, output shape: {output_shape}')

    model = load_model(input_shape, output_shape)

    pred_images, th_images = predict(X, model)

    save_path = DIR_PATH / 'prediction_result'
    save_pictures(save_path, X_image, y_images, pred_images, th_images)
