import json
from datetime import datetime
from pathlib import Path
from pprint import pformat

import numpy as np

from ..image_generator.generate import LayeredImage, generate_train_data
from ..nn.gpu import CP
from ..nn.progress_tracker import ProgressTracker
from ..nn.trainer import Trainer
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


def emit_status(status):
    if isinstance(status, str):
        emit('progress_tracker', status)
        return
    status = {
        name: {
            e['name']: {'done': e['done'], 'time': str(e['time'])}
            for e in events}
        for name, events in status.items()
    }
    emit('progress_tracker', status)


def train_model(use_gpu):
    if use_gpu:
        CP.use_gpu()
        print('Using GPU')
    else:
        CP.use_cpu()
        print('Using CPU')

    def generate_data():
        picture = generate_train_data(640, 480)
        layer_names = LayeredImage.layer_names
        X = np.array(picture['image'])
        y = np.array([np.array(picture[name])
                     for name in layer_names
                     if name != 'image'])
        y = np.moveaxis(y, 0, -1)

        X = np.reshape(X, (1, *X.shape)) / 256
        y = np.reshape(y, (1, *y.shape)) / 256

        return CP.cp.array(X), CP.cp.array(y)

    tracker = ProgressTracker(emit_status)
    tracker.reset()

    X, y = generate_data()
    input_shape, output_shape = X.shape, y.shape
    message(f'Input shape: {input_shape}, output shape: {output_shape}')
    del X, y

    model_weights_file = Path('web_app', 'components', 'my_model', 'model_weights.json')
    try:
        weights = json.load(open(model_weights_file, 'r'))
    except OSError:
        print('No model_weights.json file found')
        weights = {}

    unet = make_unet(output_shape[3])
    unet.initialize(input_shape)
    unet.init_progress_tracker(tracker)
    unet.set_weights(weights)

    message(pformat(unet.get_all_output_shapes(input_shape)))
    message(f'Count of parameters: {unet.count_parameters()}')

    trainer = Trainer(unet, generate_data, generate_data, tracker)

    message(f'[{now()}] Starting training')

    while True:
        ts = now()
        train_loss, test_loss = trainer.train_once()
        json.dump(unet.get_weights(), open(model_weights_file, 'w'))
        message(f'[{now()}] Time required: {now() - ts}\n'
                f'  Train loss: {train_loss}\n  Test loss: {test_loss}')
