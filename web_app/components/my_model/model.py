import os

import numpy as np

from ..interpreter import CropAndRotateParagraphs, CropRotateAndZoomLines
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
from .constants import OUTPUT_LAYER_NAMES, OUTPUT_LAYER_TAGS


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

    ch_count = [16, len(OUTPUT_LAYER_NAMES['monochrome'])]

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
    ch_count_end = [len(OUTPUT_LAYER_NAMES['paragraph'])]

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
    ch_count_end = [len(OUTPUT_LAYER_NAMES['line'])]

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


def make_chars(input_shape, optimizer=None):
    batch_size, height, width, in_channels = input_shape
    optimizer = Adam(lr=1e-2) if optimizer is None else optimizer

    kwargs = {
        'optimizer': optimizer,
        'trainable': True,
    }

    ch_counts = [64, 128, 256]
    n_counts = [1024, 128, len(CHARS)]

    layers = {
        'conv_block': make_conv_block(
            ch_counts, kernel_size=(5, 3), padding=(0, 1), stride=(2, 1), **kwargs),
        'fixed_width': Conv2DToBatchedFixedWidthed(8),
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

    model = wrap(
        'Chars', Model(layers=layers, relations=relations),
        loss=SoftmaxCrossEntropy())
    model.initialize(input_shape)

    return model


def make_letter_spacing(input_shape, optimizer=None):
    batch_size, height, width, in_channels = input_shape
    optimizer = Adam(lr=1e-2) if optimizer is None else optimizer

    kwargs = {
        'optimizer': optimizer,
        'trainable': True,
    }

    ch_count_downs = [
        None, [16],
    ]
    ch_count_ups = [
        None, [16],
    ]
    ch_count_end = [len(OUTPUT_LAYER_NAMES['letter_spacing'])]

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
        'LetterSpacing', Model(layers=layers, relations=relations),
        loss=SegmentationDice2D())
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


class LetterSpacingSelector(IterableSelector):
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


def make_model_system(input_shape, optimizer=None, progress_tracker=None, weights=None):
    if True:
        monochrome_label_1 = 'monochrome_y'
        monochrome_label_2 = 'cropped_1_monochrome_y'
        monochrome_label_3 = 'cropped_2_monochrome_y'
        line_label_1 = 'line_y'
        line_label_2 = 'cropped_1_line_y'

    else:
        monochrome_label_1 = 'monochrome_pred'
        monochrome_label_2 = 'cropped_1_monochrome_pred'
        monochrome_label_3 = 'cropped_2_monochrome_pred'
        line_label_1 = 'line_pred'
        line_label_2 = 'cropped_1_line_pred'

    monochrome_label_1_cpu = monochrome_label_1 + '_cpu'
    monochrome_label_2_cpu = monochrome_label_2 + '_cpu'
    monochrome_label_3_cpu = monochrome_label_3 + '_cpu'
    line_label_1_cpu = line_label_1 + '_cpu'
    line_label_2_cpu = line_label_2 + '_cpu'

    monochrome = ModelComponent(
        'Monochrome', make_monochrome(input_shape, optimizer),
        StringSelector('X', 'monochrome_y', 'monochrome_pred'))
    monochrome_output_shapes = monochrome.model.get_output_shapes(input_shape)[0]

    paragraph = ModelComponent(
        'Paragraph', make_paragraph(monochrome_output_shapes, optimizer),
        StringSelector(monochrome_label_1, 'paragraph_y', 'paragraph_pred'))
    paragraph_output_shapes = paragraph.model.get_output_shapes(monochrome_output_shapes)[0]

    move_from_gpu_1 = make_move_from_gpu_component([
        ('paragraph_y', 'paragraph_y_cpu'),
        (monochrome_label_1, monochrome_label_1_cpu),
        (line_label_1, line_label_1_cpu),
        ('letter_spacing_y', 'letter_spacing_y_cpu'),
    ])

    crop_and_rotate_paragraphs = CropAndRotateParagraphs(min(4, os.cpu_count()))

    @track_function('ParagraphCrop', 'forward', progress_tracker)
    def paragraph_crop_func(context):
        def make_subelements_divisible_by(arrays, y, x):
            return [
                [make_divisible_by(t, y, x) for t in array]
                for array in arrays
            ]
        mask, *arrays = get_from_context(context, [
            'paragraph_y_cpu',
            monochrome_label_1_cpu, line_label_1_cpu, 'letter_spacing_y_cpu'
        ])
        results = make_subelements_divisible_by(
            crop_and_rotate_paragraphs(mask, arrays), 16, 16)
        put_to_context(context, [
            monochrome_label_2_cpu, line_label_2_cpu, 'cropped_1_letter_spacing_y_cpu'
        ], results)
    paragraph_crop = RawFunctionComponent(paragraph_crop_func)

    move_to_gpu_1 = make_move_to_gpu_component([
        ('cropped_1_monochrome_y_cpu', 'cropped_1_monochrome_y'),
        (line_label_2_cpu, line_label_2),
    ])

    line = ModelComponent(
        'Line', make_line(paragraph_output_shapes, optimizer),
        LineSelector(monochrome_label_2, line_label_2, 'cropped_1_line_pred'))
    line_output_shapes = line.model.get_output_shapes(paragraph_output_shapes)[0]

    move_from_gpu_2 = make_move_from_gpu_component([
        (line_label_2, line_label_2_cpu),
    ])

    crop_rotate_and_zoom_lines = CropRotateAndZoomLines(min(4, os.cpu_count()), 32, 32)

    @track_function('LineCrop', 'forward', progress_tracker)
    def line_crop_func(context):
        def make_subelements_divisible_by(arrays, y, x):
            return [
                [
                    [make_divisible_by(line, y, x) for line in paragraph]
                    for paragraph in array
                ]
                for array in arrays
            ]

        masks, *arrays = get_from_context(context, [
            line_label_2_cpu,
            'cropped_1_monochrome_y_cpu', 'cropped_1_letter_spacing_y_cpu'
        ])

        results = make_subelements_divisible_by(
            crop_rotate_and_zoom_lines(masks, arrays), 16, 16)

        put_to_context(context, [
            monochrome_label_3_cpu, 'cropped_2_letter_spacing_y_cpu'
        ], results)
    line_crop = RawFunctionComponent(line_crop_func)

    move_to_gpu_2 = make_move_to_gpu_component([
        (monochrome_label_3_cpu, monochrome_label_3),
        ('cropped_2_letter_spacing_y_cpu', 'cropped_2_letter_spacing_y'),
    ])

    letter_spacing = ModelComponent(
        'LetterSpacing', make_letter_spacing(line_output_shapes, optimizer),
        LetterSpacingSelector(
            monochrome_label_3,
            'cropped_2_letter_spacing_y',
            'cropped_2_letter_spacing_pred'))

    import time

    @track_function('Sleep1', 'forward', progress_tracker)
    def sleep1(context):
        time.sleep(1)

    @track_function('Sleep2', 'forward', progress_tracker)
    def sleep2(context):
        time.sleep(2)

    model_system = ModelSystem([
        monochrome,
        paragraph,
        move_from_gpu_1,
        paragraph_crop,
        move_to_gpu_1,
        line,
        move_from_gpu_2,
        line_crop,
        move_to_gpu_2,
        letter_spacing,
        # RawFunctionComponent(sleep1),
        # RawFunctionComponent(sleep2),
    ])

    models = {
        'Monochrome': monochrome.model,
        'Paragraph': paragraph.model,
        'Line': line.model,
        'LetterSpacing': letter_spacing.model,
    }
    for model_name, model in models.items():
        if progress_tracker is not None:
            model.init_progress_tracker(progress_tracker, model_name)
        if weights is not None:
            model.set_weights(weights)

    names = [
        'Monochrome',
        'Paragraph',
        'ParagraphCrop',
        'Line',
        'LineCrop',
        'LetterSpacing',
        'Sleep1',
        'Sleep2',
    ]

    return model_system, models, names


def make_context(X, ys):
    context = {
        'X': X,
        **{f'{tag}_y': ys[i] for i, tag in enumerate(OUTPUT_LAYER_TAGS)},
    }
    return context
