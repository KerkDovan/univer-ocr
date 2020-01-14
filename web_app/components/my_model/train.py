import json
from pprint import pprint

import numba

from ..nn.gpu import CP
from ..nn.optimizers import Adam
from ..nn.progress_tracker import ProgressTracker
from .constants import MODEL_WEIGHTS_FILE_PATH, TRAIN_PROGRESS_PATH
from .datasets import (
    RandomSelectDataset, decode_X, decode_y, decode_ys, train_dataset, validation_dataset)
from .model import make_model_system
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


def emit_info(info):
    if emitter is None:
        for info_type, info_data in info.items():
            print(f'{info_type}:')
            pprint(info_data, indent=4)
            print()
        return
    emit('info', info)


def emit_status(status_type, status_data=None):
    if status_type in ['forward', 'backward']:
        status_type = 'forward_backward'
        status_data = {
            name: {
                e['name']: {
                    'counter': e['counter'],
                    'done': e['done'],
                    'time': str(e['time'])
                } for e in events}
            for name, events in status_data.items()
        }
    status = {'type': status_type}
    if status_data is not None:
        status['data'] = status_data
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

    model_weights_file = MODEL_WEIGHTS_FILE_PATH
    train_progress_path = TRAIN_PROGRESS_PATH

    model_systems = [
        (make_model_system, 0.0015, 0.995, 100),
    ]

    for make_model_system_func, lr, lr_step, epochs in model_systems:
        random_train_dataset = RandomSelectDataset(50, train_dataset)
        random_validation_dataset = RandomSelectDataset(5, validation_dataset)

        X, ys = random_train_dataset.get(0)
        input_shape, output_shapes = X.shape, [y.shape for y in ys]
        message(f'Input shape: {input_shape}, output shapes: {output_shapes}')
        del X, ys

        try:
            weights = json.load(open(model_weights_file, 'r'))
        except OSError:
            print('No model_weights.json file found')
            weights = {}

        optimizer = Adam(lr=lr)
        model_system, models, names = make_model_system_func(
            input_shape, optimizer, tracker, weights)

        def update_weights_func(models_to_update):
            try:
                weights = json.load(open(model_weights_file, 'r'))
            except OSError:
                weights = {}
            for name, model in models.items():
                if name not in models_to_update:
                    continue
                weights.update(model.get_weights())
            json.dump(weights, open(model_weights_file, 'w'), separators=(',', ':'))

        if save_train_progress:
            def save_pictures_func(epoch, phase, index, context):
                def save(name, X, y, pred, th, paragraph_id=None, line_id=None):
                    sp = TRAIN_PROGRESS_PATH / f'{name}'
                    sp.mkdir(parents=True, exist_ok=True)
                    prefix = f'{epoch}_{phase}_{index}_'
                    paragraph_id = '' if paragraph_id is None else f'{paragraph_id}_'
                    line_id = '' if line_id is None else f'{line_id}_'
                    for i in range(len(X)):
                        X[i].save(sp / f'{prefix}{paragraph_id}{line_id}1_{i}_1_X.png')
                    for i in range(len(y)):
                        y[i].save(sp / f'{prefix}{paragraph_id}{line_id}2_{i}_2_y.png')
                        pred[i].save(sp / f'{prefix}{paragraph_id}{line_id}2_{i}_3_pred.png')
                        th[i].save(sp / f'{prefix}{paragraph_id}{line_id}2_{i}_4_th.png')

                def delist(lst):
                    return [x[0] for x in lst]

                X = [decode_X(context['X'])]
                monochrome_y, _ = decode_y(context['monochrome_y'])
                monochrome_pred, monochrome_th = decode_ys(context['monochrome_pred'])
                save('monochrome', X, monochrome_y, monochrome_pred, monochrome_th)

                paragraph_y, _ = decode_y(context['paragraph_y'])
                paragraph_pred, paragraph_th = decode_ys(context['paragraph_pred'])
                save('paragraph', monochrome_pred, paragraph_y, paragraph_pred, paragraph_th)

                c1_m_y = context['cropped_1_monochrome_y']
                c1_l_y = context['cropped_1_line_y']
                c1_l_pred = delist(context['cropped_1_line_pred'])

                for paragraph_id in range(len(c1_m_y)):
                    line_X, _ = decode_y(c1_m_y[paragraph_id])
                    line_y, _ = decode_y(c1_l_y[paragraph_id])
                    line_pred, line_th = decode_y(c1_l_pred[paragraph_id])
                    save('line', line_X, line_y, line_pred, line_th, paragraph_id=paragraph_id)

                    c2_m_y = context['cropped_2_monochrome_y'][paragraph_id]
                    c2_ls_y = context['cropped_2_letter_spacing_y'][paragraph_id]
                    c2_ls_pred = delist(context['cropped_2_letter_spacing_pred'][paragraph_id])

                    for line_id in range(len(c2_m_y)):
                        letter_spacing_X, _ = decode_y(c2_m_y[line_id])
                        letter_spacing_y, _ = decode_y(c2_ls_y[line_id])
                        letter_spacing_pred, letter_spacing_th = decode_y(c2_ls_pred[line_id])
                        save('letter_spacing', letter_spacing_X, letter_spacing_y,
                             letter_spacing_pred, letter_spacing_th,
                             paragraph_id=paragraph_id, line_id=line_id)

            print(f'Saving train progress into {train_progress_path}\n')
        else:
            save_pictures_func = None

        layer_names = names + [
            layer_name
            for model in models.values()
            for layer_name in model.get_leaf_layers().keys()
        ]

        output_shapes = {}
        for model_name, model in models.items():
            tmp_output_shapes = model.get_all_output_shapes(input_shape)
            tmp_output_shapes = {
                model_name: tmp_output_shapes[0],
                **{name: shapes for name, shapes in tmp_output_shapes[1].items()},
            }
            for layer_name, out_shapes in tmp_output_shapes.items():
                output_shapes[layer_name] = [str(x) for x in out_shapes]

        receptive_fields = {}
        for model in models.values():
            tmp_receptive_fields = model.get_receptive_fields()
            for layer_name, rf in tmp_receptive_fields.items():
                y, x = rf['input 0']['y'], rf['input 0']['x']
                cnt = rf['input 0']['cnt']
                receptive_fields[layer_name] = f'y={y}, x={x}, size={cnt}'

        emit_info({
            'layer_names': layer_names,
            'output_shapes': output_shapes,
            'receptive_fields': receptive_fields,
        })

        count_parameters = sum(model.count_parameters() for model in models.values())
        message(f'Count of parameters: {count_parameters}')

        trainer = Trainer(
            model_system, models, random_train_dataset, random_validation_dataset,
            progress_tracker=tracker, show_progress_bar=show_progress_bar,
            optimizer=optimizer, learning_rate_step=lr_step,
            save_weights_func=update_weights_func, save_pictures_func=save_pictures_func)

        best_loss, best_loss_epoch = trainer.train(num_epochs=epochs)
        message(f'Complete. Best loss was {best_loss} on epoch #{best_loss_epoch}')
