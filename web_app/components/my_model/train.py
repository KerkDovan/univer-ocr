import json
from datetime import datetime
from pathlib import Path
from pprint import pformat

import numba

from ..nn.gpu import CP
from ..nn.progress_tracker import ProgressTracker
from ..nn.trainer import Trainer
from .model import make_unet
from .train_data import DataGenerator

emitter = None


def now():
    return datetime.now()


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


def train_model(use_gpu=False):
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

    data_generator = DataGenerator(width=16 * 40, height=16 * 30, queue_size=3)
    data_generator.start()

    def generate_data():
        X, y = data_generator.get_data()
        return CP.copy(X), CP.copy(y)

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

    model = make_unet(output_shape[3])
    model.initialize(input_shape)
    model.init_progress_tracker(tracker)
    model.set_weights(weights)

    message(pformat(model.get_all_output_shapes(input_shape)))
    message(f'Count of parameters: {model.count_parameters()}')

    trainer = Trainer(model, generate_data, generate_data, tracker)

    message(f'[{now()}] Starting training')

    cnt = 1
    while True:
        ts = now()
        train_loss, test_loss = trainer.train_once(num_epochs=100)
        json.dump(model.get_weights(), open(model_weights_file, 'w'))
        message(f'[{now()}] Time required: {now() - ts} #{cnt}\n\n')
        cnt += 1
