import json
from pathlib import Path
from pprint import pformat

import numba

from ..nn.gpu import CP
from ..nn.optimizers import Adam
from ..nn.progress_tracker import ProgressTracker
from .datasets import (
    RandomSelectDataset, decode_X, decode_y, save_pictures, train_dataset, validation_dataset)
from .model import make_unet
from .trainer import Trainer

emitter = None


def init_emitter(new_emitter):
    global emitter
    emitter = new_emitter


def emit(message_type, obj):
    if emitter is None:
        return
    emitter.emit(message_type, obj)


def message(*message, sep=' ', end='\n'):
    text = sep.join(str(x) for x in message) + end
    if emitter is None:
        print(text)
        return
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


def train_model(use_gpu=False, show_progress_bar=False, save_train_progress=False):
    if use_gpu:
        CP.use_gpu()
        print('Using GPU')
        gpu = numba.cuda.get_current_device()
        print(
            f'name = {gpu.name}\n'
            f'maxThreadsPerBlock = {gpu.MAX_THREADS_PER_BLOCK}\n'
            f'maxBlockDimX = {gpu.MAX_BLOCK_DIM_X}\n'
            f'maxBlockDimY = {gpu.MAX_BLOCK_DIM_Y}\n'
            f'maxBlockDimZ = {gpu.MAX_BLOCK_DIM_Z}\n'
            f'maxGridDimX = {gpu.MAX_GRID_DIM_X}\n'
            f'maxGridDimY = {gpu.MAX_GRID_DIM_Y}\n'
            f'maxGridDimZ = {gpu.MAX_GRID_DIM_Z}\n'
            f'maxSharedMemoryPerBlock = {gpu.MAX_SHARED_MEMORY_PER_BLOCK}\n'
            f'asyncEngineCount = {gpu.ASYNC_ENGINE_COUNT}\n'
            f'canMapHostMemory = {gpu.CAN_MAP_HOST_MEMORY}\n'
            f'multiProcessorCount = {gpu.MULTIPROCESSOR_COUNT}\n'
            f'warpSize = {gpu.WARP_SIZE}\n'
            f'unifiedAddressing = {gpu.UNIFIED_ADDRESSING}\n'
            f'pciBusID = {gpu.PCI_BUS_ID}\n'
            f'pciDeviceID = {gpu.PCI_DEVICE_ID}\n'
        )
    else:
        CP.use_cpu()
        print('Using CPU')

    tracker = ProgressTracker(emit_status)
    tracker.reset()

    model_weights_file = Path('web_app', 'components', 'my_model', 'model_weights.json')
    train_progress_path = Path('web_app', 'components', 'my_model', 'train_progress')

    try:
        weights = json.load(open(model_weights_file, 'r'))
    except OSError:
        print('No model_weights.json file found')
        weights = {}

    random_train_dataset = RandomSelectDataset(10, train_dataset)
    random_validation_dataset = RandomSelectDataset(1, validation_dataset)

    X, y = random_train_dataset.get(0)
    input_shape, output_shape = X.shape, y.shape
    message(f'Input shape: {input_shape}, output shape: {output_shape}')
    del X, y

    optimizer = Adam(lr=0.011)
    model = make_unet(input_shape[3], output_shape[3], optimizer)
    model.initialize(input_shape)
    model.init_progress_tracker(tracker)
    model.set_weights(weights)

    def load_weights_func():
        model.set_weights(json.load(open(model_weights_file, 'r')))

    def save_weights_func():
        json.dump(model.get_weights(), open(model_weights_file, 'w'), separators=(',', ':'))

    if save_train_progress:
        def save_pictures_func(epoch, phase, index, X, y, p):
            X_image = decode_X(X)
            y_images, _ = decode_y(y)
            pred_images, th_images = decode_y(p)
            save_pictures(train_progress_path, X_image, y_images, pred_images, th_images,
                          f'{epoch}_{phase}_{index}')
        print(f'Saving train progress into {train_progress_path}\n')
    else:
        save_pictures_func = None

    message(pformat(model.get_all_output_shapes(input_shape)))
    message(f'Count of parameters: {model.count_parameters()}')

    trainer = Trainer(
        model, random_train_dataset, random_validation_dataset,
        progress_tracker=tracker, show_progress_bar=show_progress_bar,
        optimizer=optimizer, learning_rate_step=0.995,
        save_weights_func=save_weights_func, save_pictures_func=save_pictures_func)

    best_loss, best_loss_epoch = trainer.train(num_epochs=1000)
    message(f'Complete. Best loss was {best_loss} on epoch #{best_loss_epoch}')
