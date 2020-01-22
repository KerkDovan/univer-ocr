import json
import random

from PIL import Image

from ..nn.gpu import CP
from .constants import MODEL_WEIGHTS_FILE_PATH, PREDICTION_RESULT_PATH, PREDICTION_SOURCE_PATH
from .datasets import encode_X, validation_dataset
from .model import make_divisible_by, make_model_system


def load_model_system(input_shape):
    model_weights_file = MODEL_WEIGHTS_FILE_PATH
    try:
        weights = json.load(open(model_weights_file, 'r'))
    except OSError:
        print('No model_weights.json file found')
        weights = {}

    model_system, models, *_ = make_model_system(input_shape)
    for model in models.values():
        model.set_weights(weights)
    return model_system


def main(use_gpu=False, filename=None):
    if use_gpu:
        CP.use_gpu()
        print('Using GPU')
    else:
        CP.use_cpu()
        print('Using CPU')

    if filename is None:
        dataset = validation_dataset
        print('Using validation dataset')

        idx = random.randint(0, len(dataset) - 1)
        print(f'Data #{idx}')

        layer_images = dataset.get_images(idx, ['image'])
        X_image = layer_images['image']

    else:
        print(f'Using file {filename}')
        X_image = Image.open(PREDICTION_SOURCE_PATH / filename)

    X = encode_X(X_image.convert('L'))
    X = make_divisible_by(X, 16, 16)
    context = {}
    context['monochrome_X'] = X

    input_shape = X.shape
    print(f'Input shape: {input_shape}')

    model_system = load_model_system(input_shape)
    model_system.predict(context)

    pred_text = context['text']

    save_path = PREDICTION_RESULT_PATH
    save_path.mkdir(parents=True, exist_ok=True)
    X_image.save(save_path / 'X.png')

    with open(save_path / 'result.txt', 'w') as fp:
        print(pred_text, file=fp)
