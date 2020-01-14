from ..nn.help_func import make_list_if_not
from ..nn.layers import Concat, Convolutional2D, LeakyRelu, Sigmoid, Upsample2D
from ..nn.losses import SegmentationDice2D
from ..nn.model_system import FunctionComponent, ModelComponent, ModelSystem
from ..nn.models import Model
from ..nn.optimizers import Adam
from ..nn.regularizations import L2
from .constants import OUTPUT_LAYER_NAMES, OUTPUT_LAYER_TAGS


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


def make_monochrome(input_shape, optimizer=None):
    batch_size, height, width, in_channels = input_shape
    optimizer = Adam(lr=1e-2) if optimizer is None else optimizer

    kwargs = {
        'optimizer': optimizer,
        'trainable': True,
    }

    ch_count = [len(OUTPUT_LAYER_NAMES['monochrome'])]

    layers = {
        'Monochrome': make_conv_block(
            ch_count, last_sigmoid=True,
            kernel_size=(5, 5), padding=2, **kwargs),
    }
    relations = {
        'Monochrome': 0,
        0: 'Monochrome',
    }

    model = Model(
        layers=layers, relations=relations,
        loss=SegmentationDice2D())

    return model


def make_paragraph(input_shape, optimizer=None):
    batch_size, height, width, in_channels = input_shape
    optimizer = Adam(lr=1e-2) if optimizer is None else optimizer

    kwargs = {
        'optimizer': optimizer,
        'trainable': True,
    }

    ch_count_downs = [
        None, [1], [1], [1], [1],
    ]
    ch_count_ups = [
        None, [1], [1], [1], [1],
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
            ch_count_end,
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

    return model


def make_model_system(input_shape, optimizer=None, progress_tracker=None, weights=None):
    monochrome = ModelComponent(
        'Monochrome', make_monochrome(input_shape, optimizer),
        'X', 'monochrome_y', 'monochrome_pred')
    monochrome_output_shapes = monochrome.model.get_output_shapes(input_shape)[0]

    paragraph = ModelComponent(
        'Paragraph', make_paragraph(monochrome_output_shapes, optimizer),
        'monochrome_pred', 'paragraph_y', 'paragraph_pred')

    model_system = ModelSystem([
        monochrome,
        paragraph,
    ])

    models = {
        'Monochrome': monochrome.model,
        'Paragraph': paragraph.model,
    }
    for model_name, model in models.items():
        model.initialize(input_shape)
        if progress_tracker is not None:
            model.init_progress_tracker(progress_tracker, model_name)
        if weights is not None:
            model.set_weights(weights)

    names = [
        'Monochrome',
        'Paragraph',
    ]

    return model_system, models, names


def make_context(X, ys):
    context = {
        'X': X,
        **{f'{tag}_y': ys[i] for i, tag in enumerate(OUTPUT_LAYER_TAGS)},
    }
    return context
