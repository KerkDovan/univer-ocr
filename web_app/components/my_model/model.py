from ..nn.help_func import make_list_if_not
from ..nn.layers import Concat, Convolutional2D, LeakyRelu, Sigmoid, Upsample2D
from ..nn.losses import SegmentationDice2D
from ..nn.models import Model
from ..nn.optimizers import Adam
from ..nn.regularizations import L2
from .constants import OUTPUT_LAYER_NAMES, OUTPUT_LAYER_TAGS, OUTPUT_LAYER_TAGS_IDS


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


def make_start(input_shape, optimizer=None):
    batch_size, height, width, in_channels = input_shape
    optimizer = Adam(lr=1e-2) if optimizer is None else optimizer

    kwargs = {
        'optimizer': optimizer,
        'trainable': True,
    }

    ch_count_start = [len(OUTPUT_LAYER_NAMES['monochrome'])]

    model = Model(layers={
        'start': make_conv_block(
            ch_count_start, last_sigmoid=True,
            kernel_size=(5, 5), padding=2, **kwargs),
    }, relations={
        'start': 0,
        0: 'start',
    }, loss=SegmentationDice2D())

    return model


def make_model(input_shape, optimizer=None):
    batch_size, height, width, in_channels = input_shape
    optimizer = Adam(lr=1e-2) if optimizer is None else optimizer

    kwargs = {
        'optimizer': optimizer,
        'trainable': True,
    }

    layer_tags = [
        layer_tag for layer_tag in OUTPUT_LAYER_TAGS
        if layer_tag != 'monochrome'
    ]

    ch_count_start = [len(OUTPUT_LAYER_NAMES['monochrome'])]
    ch_count_downs = [
        [64], [64],
    ]
    ch_count_down_bottom = [128]
    ch_count_bottom = [64, 32, 16, 8]
    ch_count_ups = [
        [32, 16, 8], [32, 16, 8],
    ]
    ch_count_up_end = [8]
    ch_count_ends = {
        layer_tag: [len(OUTPUT_LAYER_NAMES[layer_tag])]
        for layer_tag in layer_tags
    }

    assert len(ch_count_downs) == len(ch_count_ups) > 0
    depth = len(ch_count_downs)

    layers = {
        'start': make_conv_block(
            ch_count_start, last_sigmoid=True,
            kernel_size=(5, 5), padding=2, **kwargs),
        **{
            f'down_{i + 1}': make_conv_block(
                ch_count_downs[i], kernel_size=(3, 3), padding=1, stride=2, **kwargs)
            for i in range(depth)
        },
        'down_bottom': make_conv_block(
            ch_count_down_bottom, kernel_size=(3, 3), padding=1, stride=2, **kwargs),
        'bottom': make_conv_block(
            ch_count_bottom, kernel_size=(3, 3), padding=1, **kwargs),
        **{
            f'up_{i + 1}': make_up(
                ch_count_ups[i], kernel_size=(3, 3), padding=1, **kwargs)
            for i in range(depth)
        },
        'up_end': make_up(
            ch_count_up_end, kernel_size=(3, 3), padding=1, **kwargs),
        **{
            f'end_{layer_tag}': make_conv_block(
                ch_count_ends[layer_tag], last_sigmoid=True,
                kernel_size=(3, 3), padding=1, **kwargs)
            for layer_tag in layer_tags
        }
    }

    relations = {
        'start': 0,
        0: 'start',
        'down_1': 'start',
        **{
            f'down_{i + 1}': f'down_{i}'
            for i in range(1, depth)
        },
        'down_bottom': f'down_{depth}',
        'bottom': 'down_bottom',
        f'up_{depth}': [f'down_{depth}', 'bottom'],
        **{
            f'up_{i}': [f'down_{i}', f'up_{i + 1}']
            for i in range(1, depth)
        },
        'up_end': ['start', 'up_1'],
        **{
            f'end_{layer_tag}': 'up_end'
            for layer_tag in layer_tags
        },
        **{
            OUTPUT_LAYER_TAGS_IDS[layer_tag]: f'end_{layer_tag}'
            for layer_tag in layer_tags
        },
    }

    return Model(layers=layers, relations=relations, loss=SegmentationDice2D())
