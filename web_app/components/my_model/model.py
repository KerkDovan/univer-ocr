import os
from enum import Enum

import numpy as np

from ..interpreter import CropAndRotateParagraphs, CropRotateAndZoomLines, LabelChar, PredToText
from ..nn.gpu import CP
from ..nn.help_func import make_list_if_not
from ..nn.layers import (
    Concat, Conv2DToBatchedFixedWidthed, Convolutional2D, Flatten, FullyConnected, LeakyRelu,
    Sigmoid, Upsample2D)
from ..nn.losses import SegmentationDice2D, SoftmaxCrossEntropy
from ..nn.model_system import (
    IterableSelector, ModelComponent, ModelSystem, RawFunctionComponent, StringSelector)
from ..nn.models import Model
from ..nn.optimizers import Adam
from ..nn.progress_tracker import track_function
from ..nn.regularizations import L2
from ..primitives import CHARS
from .constants import LAYER_NAMES

CHAR_INPUT_HEIGHT = 32
CHAR_FIXED_WIDTH = 8


def make_divisible_by(arr, y, x):
    b, h, w, c = arr.shape
    to_add_y = y - h % y
    to_add_x = x - w % x
    py, px = to_add_y // 2, to_add_x // 2
    new_shape = (b, h + to_add_y, w + to_add_x, c)
    new_arr = np.zeros(new_shape)
    new_arr[:, py:py + h, px:px + w, :] = arr
    return new_arr


def make_conv(out_ch, kernel_size=(5, 5), padding=2, **kwargs):
    return Convolutional2D(kernel_size, out_channels=out_ch, padding=padding,
                           regularizer=L2(0.01), **kwargs)


def make_conv_block(out_chs, last_sigmoid=False, **kwargs):
    out_chs = make_list_if_not(out_chs)
    layers = {}
    relations = {}
    prev = 0
    for i in range(1, len(out_chs) + 1):
        conv_name, conv = f'conv_{i}', make_conv(out_chs[i - 1], **kwargs)
        layers[conv_name] = conv
        if i == len(out_chs) and last_sigmoid is True:
            activation_name, activation = f'sigmoid', Sigmoid()
        else:
            activation_name, activation = f'leaky_relu_{i}', LeakyRelu(0.01)
        layers[activation_name] = activation
        relations[conv_name] = prev
        relations[activation_name] = conv_name
        prev = activation_name
    relations[0] = prev
    return Model(layers, relations)


def make_up(out_chs, **kwargs):
    return Model(layers={
        'upsample': Upsample2D(2),
        'concat': Concat(),
        'conv_block': make_conv_block(out_chs, **kwargs),
    }, relations={
        'upsample': 1,
        'concat': ['upsample', 0],
        'conv_block': 'concat',
        0: 'conv_block',
    })


def make_single_up(out_chs, **kwargs):
    return Model(layers={
        'upsample': Upsample2D(2),
        'conv_block': make_conv_block(out_chs, **kwargs),
    }, relations={
        'upsample': 0,
        'conv_block': 'upsample',
        0: 'conv_block',
    })


def wrap(name, model, **kwargs):
    return Model(layers={name: model}, relations={name: 0, 0: name}, **kwargs)


def make_edge_detection(input_shape):
    batch_size, height, width, in_channels = input_shape
    w = np.expand_dims(np.expand_dims(np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0],
    ]), 0), -1)
    b = np.zeros((in_channels,))
    conv = Convolutional2D(
        (3, 3), in_channels=in_channels, out_channels=in_channels,
        padding=1, w=w, b=b, trainable=False)

    def func(X):
        return conv.forward(X)[0]

    return func


def make_monochrome(input_shape, optimizer=None):
    batch_size, height, width, in_channels = input_shape
    optimizer = Adam(lr=1e-2) if optimizer is None else optimizer

    kwargs = {
        'optimizer': optimizer,
        'trainable': True,
    }

    ch_count = [16, len(LAYER_NAMES['monochrome'])]

    layers = {
        'Monochrome': make_conv_block(
            ch_count, last_sigmoid=True,
            kernel_size=(3, 3), padding=1, **kwargs),
    }
    relations = {
        'Monochrome': 0,
        0: 'Monochrome',
    }

    model = Model(
        layers=layers, relations=relations,
        loss=SegmentationDice2D())
    model.initialize(input_shape)

    return model


def make_paragraph(input_shape, optimizer=None):
    batch_size, height, width, in_channels = input_shape
    optimizer = Adam(lr=1e-2) if optimizer is None else optimizer

    kwargs = {
        'optimizer': optimizer,
        'trainable': True,
    }

    ch_count_downs = [
        None, [1], [1],
    ]
    ch_count_ups = [
        None, [1], [1],
    ]
    ch_count_end = [len(LAYER_NAMES['paragraph'])]

    layers = {
        **{
            f'down_{i}': make_conv_block(
                ch_count_downs[i],
                kernel_size=(5, 5), padding=2, stride=2, **kwargs)
            for i in range(1, len(ch_count_downs))
        },
        **{
            f'up_{i}': make_single_up(
                ch_count_ups[i],
                kernel_size=(5, 5), padding=2, **kwargs)
            for i in range(1, len(ch_count_ups))
        },
        'end': make_conv_block(
            ch_count_end, last_sigmoid=True,
            kernel_size=(5, 5), padding=2, **kwargs),
    }
    relations = {
        'down_1': 0,
        **{
            f'down_{i + 1}': f'down_{i}'
            for i in range(1, len(ch_count_downs) - 1)
        },
        f'up_{len(ch_count_ups) - 1}': f'down_{len(ch_count_downs) - 1}',
        **{
            f'up_{i}': f'up_{i + 1}'
            for i in range(1, len(ch_count_ups) - 1)
        },
        'end': 'up_1',
        0: 'end',
    }

    model = wrap(
        'Paragraph', Model(layers=layers, relations=relations),
        loss=SegmentationDice2D())
    model.initialize(input_shape)

    return model


def make_line(input_shape, optimizer=None):
    batch_size, height, width, in_channels = input_shape
    optimizer = Adam(lr=1e-2) if optimizer is None else optimizer

    kwargs = {
        'optimizer': optimizer,
        'trainable': True,
    }

    ch_count_downs = [
        None, [4], [4],
    ]
    ch_count_ups = [
        None, [4], [4],
    ]
    ch_count_end = [len(LAYER_NAMES['line'])]

    layers = {
        **{
            f'down_{i}': make_conv_block(
                ch_count_downs[i],
                kernel_size=(5, 5), padding=2, stride=2, **kwargs)
            for i in range(1, len(ch_count_downs))
        },
        **{
            f'up_{i}': make_single_up(
                ch_count_ups[i],
                kernel_size=(5, 5), padding=2, **kwargs)
            for i in range(1, len(ch_count_ups))
        },
        'end': make_conv_block(
            ch_count_end, last_sigmoid=True,
            kernel_size=(5, 5), padding=2, **kwargs),
    }
    relations = {
        'down_1': 0,
        **{
            f'down_{i + 1}': f'down_{i}'
            for i in range(1, len(ch_count_downs) - 1)
        },
        f'up_{len(ch_count_ups) - 1}': f'down_{len(ch_count_downs) - 1}',
        **{
            f'up_{i}': f'up_{i + 1}'
            for i in range(1, len(ch_count_ups) - 1)
        },
        'end': 'up_1',
        0: 'end',
    }

    model = wrap(
        'Line', Model(layers=layers, relations=relations),
        loss=SegmentationDice2D())
    model.initialize(input_shape)

    return model


def make_dense_block(out_counts, **kwargs):
    out_counts = make_list_if_not(out_counts)
    layers = {}
    relations = {}
    prev = 0
    for i in range(1, len(out_counts) + 1):
        dense_name, dense = f'dense_{i}', FullyConnected(n_output=out_counts[i - 1], **kwargs)
        layers[dense_name] = dense
        relations[dense_name] = prev
        if i < len(out_counts):
            activation_name, activation = f'leaky_relu_{i}', LeakyRelu(0.01)
            layers[activation_name] = activation
            relations[activation_name] = dense_name
            prev = activation_name
        else:
            prev = dense_name
    relations[0] = prev
    return Model(layers, relations)


def make_char(input_shape, optimizer=None):
    batch_size, _, width, in_channels = input_shape
    optimizer = Adam(lr=1e-2) if optimizer is None else optimizer

    kwargs = {
        'optimizer': optimizer,
        'trainable': True,
    }

    ch_counts = [64, 64, 64]
    n_counts = [1024, 128, len(CHARS)]

    layers = {
        'conv_block': make_conv_block(
            ch_counts, kernel_size=(5, 3), padding=(0, 1), stride=(2, 1), **kwargs),
        'fixed_width': Conv2DToBatchedFixedWidthed(CHAR_FIXED_WIDTH),
        'flatten': Flatten(),
        'dense_block': make_dense_block(n_counts, **kwargs),
    }
    relations = {
        'conv_block': 0,
        'fixed_width': 'conv_block',
        'flatten': 'fixed_width',
        'dense_block': 'flatten',
        0: 'dense_block'
    }

    input_shape = (batch_size, CHAR_INPUT_HEIGHT, width, in_channels)
    model = wrap(
        'Char', Model(layers=layers, relations=relations),
        loss=SoftmaxCrossEntropy())
    model.initialize(input_shape)

    return model


def make_move_from_gpu_component(labels):
    def move_from_gpu(var):
        if isinstance(var, list):
            return [move_from_gpu(v) for v in var]
        if isinstance(var, dict):
            return {k: move_from_gpu(v) for k, v in var.items()}
        return CP.asnumpy(var)

    def func(context):
        for old_label, new_label in labels:
            context[new_label] = move_from_gpu(context[old_label])

    return RawFunctionComponent(func)


def make_move_to_gpu_component(labels):
    def move_to_gpu(var):
        if isinstance(var, list):
            return [move_to_gpu(v) for v in var]
        if isinstance(var, dict):
            return {k: move_to_gpu(v) for k, v in var.items()}
        return CP.copy(var)

    def func(context):
        for old_label, new_label in labels:
            context[new_label] = move_to_gpu(context[old_label])

    return RawFunctionComponent(func)


def get_from_context(context, labels):
    return [context[label] for label in labels]


def put_to_context(context, labels, values):
    for label, value in zip(labels, values):
        context[label] = value


def make_rename_in_context_component(labels):
    def rename_in_context(context):
        for old_label, new_label in labels:
            context[new_label] = context[old_label]
    return RawFunctionComponent(rename_in_context)


class LineSelector(IterableSelector):
    def __init__(self, X_label, y_label, pred_label):
        super().__init__(X_label, y_label, pred_label)
        self.paragraph_id = 0

    def __call__(self, context):
        super().__call__(context)
        self.paragraph_id = 0

    def get(self):
        for i in range(len(self.context[self.X_label])):
            self.paragraph_id = i
            yield self.context[self.X_label][i], self.context[self.y_label][i]

    def put(self, pred):
        if self.pred_label not in self.context.keys():
            self.context[self.pred_label] = []
        if self.paragraph_id >= len(self.context[self.pred_label]):
            self.context[self.pred_label].append([])
        self.context[self.pred_label][self.paragraph_id] = pred


class CharSelector(IterableSelector):
    def __init__(self, X_label, y_label, pred_label):
        super().__init__(X_label, y_label, pred_label)
        self.paragraph_id = 0
        self.line_id = 0

    def __call__(self, context):
        super().__call__(context)
        self.paragraph_id = 0
        self.line_id = 0

    def get(self):
        for i in range(len(self.context[self.X_label])):
            self.paragraph_id = i
            for j in range(len(self.context[self.X_label][i])):
                self.line_id = j
                yield self.context[self.X_label][i][j], self.context[self.y_label][i][j]

    def put(self, pred):
        if self.pred_label not in self.context.keys():
            self.context[self.pred_label] = []
        if self.paragraph_id >= len(self.context[self.pred_label]):
            self.context[self.pred_label].append([])
        if self.line_id >= len(self.context[self.pred_label][self.paragraph_id]):
            self.context[self.pred_label][self.paragraph_id].append([])
        self.context[self.pred_label][self.paragraph_id][self.line_id] = pred


class Modes(Enum):
    TRAIN_MONOCHROME = 0
    TRAIN_PARAGRAPH = 1
    TRAIN_LINE = 2
    TRAIN_CHAR = 3
    TRAIN_ALL = 4
    PREDICT = 5


def make_context_maker(mode):
    def to_gpu(arr):
        if CP.is_gpu_used:
            return CP.copy(arr)
        return arr

    if mode is Modes.TRAIN_MONOCHROME:
        def make_context(dataset_get_func, args=(), kwargs={}):
            layer_tags = ['image', 'monochrome']
            layers = dataset_get_func(*args, layer_tags=layer_tags, **kwargs)
            context = {
                'monochrome_X': to_gpu(layers['image']),
                'monochrome_y': to_gpu(layers['monochrome']),
            }
            return context

    elif mode is Modes.TRAIN_PARAGRAPH:
        def make_context(dataset_get_func, args=(), kwargs={}):
            layer_tags = ['monochrome', 'paragraph']
            layers = dataset_get_func(*args, layer_tags=layer_tags, **kwargs)
            context = {
                'paragraph_X': to_gpu(layers['monochrome']),
                'paragraph_y': to_gpu(layers['paragraph']),
            }
            return context

    elif mode is Modes.TRAIN_LINE:
        def make_context(dataset_get_func, args=(), kwargs={}):
            layer_tags = ['monochrome', 'paragraph', 'line']
            layers = dataset_get_func(*args, layer_tags=layer_tags, **kwargs)
            context = {
                'monochrome_pred_cpu': layers['monochrome'],
                'paragraph_pred_cpu': layers['paragraph'],
                'line_cpu': layers['line'],
            }
            return context

    elif mode is Modes.TRAIN_CHAR:
        def make_context(dataset_get_func, args=(), kwargs={}):
            layer_tags = ['monochrome', 'paragraph', 'line', 'char']
            layers = dataset_get_func(*args, layer_tags=layer_tags, **kwargs)
            context = {
                'monochrome_pred_cpu': layers['monochrome'],
                'paragraph_pred_cpu': layers['paragraph'],
                'line_cpu': layers['line'],
                'char_cpu': layers['char'],
            }
            return context

    elif mode is Modes.TRAIN_ALL:
        def make_context(dataset_get_func, args=(), kwargs={}):
            layer_tags = ['image', 'monochrome', 'paragraph', 'line', 'char']
            layers = dataset_get_func(*args, layer_tags=layer_tags, **kwargs)
            context = {
                'monochrome_X': to_gpu(layers['image']),
                'monochrome_y': to_gpu(layers['monochrome']),
                'paragraph_y': to_gpu(layers['paragraph']),
                'line_cpu': layers['line'],
                'char_cpu': layers['char'],
            }
            return context

    else:
        def make_context(dataset_get_func, args=(), kwargs={}):
            layer_tags = ['image']
            layers = dataset_get_func(*args, layer_tags=layer_tags, **kwargs)
            context = {
                'monochrome_X': to_gpu(layers['image']),
            }
            return context

    return make_context


def make_model_system(input_shape, optimizer=None, progress_tracker=None, weights=None,
                      mode=Modes.PREDICT):
    def get_result(components):
        order = [
            'Monochrome',
            'Paragraph', 'move_from_gpu_paragraph',
            'ParagraphCrop', 'move_to_gpu_paragraph_crop', 'rename_line',
            'Line', 'move_from_gpu_line',
            'LineCrop',
            'CharLabel', 'move_to_gpu_char_label',
            'Char', 'move_from_gpu_char',
            'PredToText',
            'Sleep1', 'Sleep2',
        ]
        model_system = ModelSystem([
            components[component_name]
            for component_name in order
            if component_name in components.keys()
        ])
        models = {
            model_name: components[model_name].model
            for model_name in ['Monochrome', 'Paragraph', 'Line', 'Char']
            if model_name in components.keys()
        }
        for model_name, model in models.items():
            if progress_tracker is not None:
                model.init_progress_tracker(progress_tracker, model_name)
            if weights is not None:
                model.set_weights(weights)
        names = [
            component_name
            for component_name in order
            if component_name in [
                'Monochrome',
                'Paragraph',
                'ParagraphCrop',
                'Line',
                'LineCrop',
                'CharLabel',
                'Char',
                'PredToText',
                'Sleep1', 'Sleep2',
            ] and component_name in components.keys()
        ]
        return model_system, models, names

    def make_monochrome_component():
        monochrome = ModelComponent(
            'Monochrome', make_monochrome(input_shape, optimizer),
            StringSelector('monochrome_X', 'monochrome_y', 'monochrome_pred'))
        return monochrome

    if mode is Modes.TRAIN_MONOCHROME:
        return get_result({'Monochrome': make_monochrome_component()})

    def make_paragraph_component():
        paragraph = ModelComponent(
            'Paragraph', make_paragraph(input_shape, optimizer),
            StringSelector('paragraph_X', 'paragraph_y', 'paragraph_pred'))
        return paragraph

    if mode is Modes.TRAIN_PARAGRAPH:
        return get_result({'Paragraph': make_paragraph_component()})

    def make_paragraph_crop_component():
        crop_and_rotate_paragraphs = CropAndRotateParagraphs(min(4, os.cpu_count()))

        @track_function('ParagraphCrop', 'forward', progress_tracker)
        def paragraph_crop_func(context):
            def make_subelements_divisible_by(arrays, y, x):
                return [
                    [make_divisible_by(t, y, x) for t in array]
                    for array in arrays
                ]
            mask, *arrays = get_from_context(context, [
                'paragraph_pred_cpu',
                'monochrome_pred_cpu', 'line_cpu', 'char_cpu',
            ])
            results = make_subelements_divisible_by(
                crop_and_rotate_paragraphs(mask, arrays), 16, 16)
            put_to_context(context, [
                'cropped_monochrome_cpu', 'cropped_line_cpu', 'cropped_char_cpu',
            ], results)
        paragraph_crop = RawFunctionComponent(paragraph_crop_func)
        return paragraph_crop

    def make_line_component():
        line = ModelComponent(
            'Line', make_line(input_shape, optimizer),
            LineSelector('cropped_monochrome', 'cropped_line', 'line_pred'))
        return line

    if mode is Modes.TRAIN_LINE:
        return get_result({
            'ParagraphCrop': make_paragraph_crop_component(),
            'move_to_gpu_paragraph_crop': make_move_to_gpu_component([
                ('cropped_monochrome_cpu', 'cropped_monochrome'),
                ('cropped_line_cpu', 'cropped_line'),
            ]),
            'Line': make_line_component(),
        })

    def make_line_crop_component():
        crop_rotate_and_zoom_lines = CropRotateAndZoomLines(
            min(8, os.cpu_count()),
            CHAR_INPUT_HEIGHT, CHAR_FIXED_WIDTH)

        @track_function('LineCrop', 'forward', progress_tracker)
        def line_crop_func(context):
            masks, *arrays = get_from_context(context, [
                'line_pred_cpu',
                'cropped_monochrome_cpu', 'cropped_char_cpu',
            ])

            results = crop_rotate_and_zoom_lines(masks, arrays)

            put_to_context(context, [
                'cropped_2_monochrome_cpu', 'cropped_2_char_cpu',
            ], results)
        line_crop = RawFunctionComponent(line_crop_func)
        return line_crop

    def make_char_label_component():
        label_char = LabelChar(min(8, os.cpu_count()))

        @track_function('CharLabel', 'forward', progress_tracker)
        def char_label_func(context):
            lines = get_from_context(context, ['cropped_2_char_cpu'])[0]
            result = label_char(lines)
            put_to_context(context, ['char_labels_cpu'], [result])
        char_label = RawFunctionComponent(char_label_func)
        return char_label

    def make_char_component():
        char = ModelComponent(
            'Char', make_char(input_shape, optimizer),
            CharSelector('cropped_2_monochrome', 'char_labels', 'char_pred'))
        return char

    if mode is Modes.TRAIN_CHAR:
        return get_result({
            'ParagraphCrop': make_paragraph_crop_component(),
            'rename_line': make_rename_in_context_component([
                ('cropped_line_cpu', 'line_pred_cpu'),
            ]),
            'LineCrop': make_line_crop_component(),
            'CharLabel': make_char_label_component(),
            'move_to_gpu_char_label': make_move_to_gpu_component([
                ('cropped_2_monochrome_cpu', 'cropped_2_monochrome'),
                ('char_labels_cpu', 'char_labels'),
            ]),
            'Char': make_char_component(),
        })

    def make_pred_to_text_component():
        pred_to_text = PredToText(min(8, os.cpu_count()))

        @track_function('PredToText', 'forward', progress_tracker)
        def pred_to_text_func(context):
            predictions = get_from_context(context, ['char_pred_cpu'])[0]
            result = pred_to_text(predictions)
            put_to_context(context, ['text'], [result])
        pred_to_text = RawFunctionComponent(pred_to_text_func)
        return pred_to_text

    if mode in [Modes.TRAIN_ALL, Modes.PREDICT]:
        components = {
            'Monochrome': make_monochrome_component(),
            'Paragraph': make_paragraph_component(),
            'move_from_gpu_paragraph': make_move_from_gpu_component([
                ('monochrome_pred', 'monochrome_pred_cpu'),
                ('paragraph_pred', 'paragraph_pred_cpu'),
            ]),
            'ParagraphCrop': make_paragraph_crop_component(),
            'move_to_gpu_paragraph_crop': make_move_to_gpu_component([
                ('cropped_monochrome_cpu', 'cropped_monochrome'),
                ('cropped_line_cpu', 'cropped_line'),
            ]),
            'Line': make_line_component(),
            'move_from_gpu_line': make_move_from_gpu_component([
                ('line_pred', 'line_pred_cpu'),
            ]),
            'LineCrop': make_line_crop_component(),
            'CharLabel': make_char_label_component(),
            'move_to_gpu_char_label': make_move_to_gpu_component([
                ('cropped_2_monochrome_cpu', 'cropped_2_monochrome'),
                ('char_labels_cpu', 'char_labels'),
            ]),
            'Char': make_char_component(),
        }
        if mode is Modes.PREDICT:
            components.update({
                'move_from_gpu_char': make_move_from_gpu_component([
                    ('char_pred', 'char_pred_cpu'),
                ]),
                'PredToText': make_pred_to_text_component(),
            })
        return get_result(components)

    import time

    def make_sleep_components():
        @track_function('Sleep1', 'forward', progress_tracker)
        def sleep1(context):
            time.sleep(1)

        @track_function('Sleep2', 'forward', progress_tracker)
        def sleep2(context):
            time.sleep(2)

        return sleep1, sleep2
