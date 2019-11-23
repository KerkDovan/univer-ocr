from datetime import datetime
from pprint import pformat

import numpy as np

from ..image_generator.generate import LayeredImage, generate_train_data
from .model import make_unet

emitter = None


def now():
    return datetime.now()


def init_emitter(new_emitter):
    global emitter
    emitter = new_emitter


def emit(message_type, obj):
    emitter.emit(message_type, obj)


def message(*message, sep=' ', end='\n'):
    text = sep.join(str(x) for x in message) + end
    emit('message', text)


def train_model():
    picture = generate_train_data(480, 640)
    layer_names = LayeredImage.layer_names
    X = np.array(picture['image'])
    y = np.array([np.array(picture[name])
                  for name in layer_names
                  if name != 'image'])
    y = np.moveaxis(y, 0, -1)

    X = np.reshape(X, (1, *X.shape)) / 256
    y = np.reshape(y, (1, *y.shape)) / 256

    message('\n\nGenerated data')

    input_shape = X.shape

    message(f'Input shape: {input_shape}, output shape: {y.shape}')

    unet = make_unet(y.shape[3])
    unet.initialize(input_shape)

    message(pformat(unet.get_all_output_shapes(input_shape)))
    message(f'Count of parameters: {unet.count_parameters()}')

    ts = now()
    loss = unet.compute_loss_and_gradients(X, y)
    message(loss)

    message(f'Complete! Time required: {now() - ts}')
