from ..nn.help_func import make_list_if_not
from ..nn.layers import Concat, Convolutional2D, LeakyRelu, MaxPool2D, Sigmoid, Upsample2D
from ..nn.losses import SegmentationDice2D
from ..nn.models import Model
from ..nn.optimizers import Adam
from ..nn.regularizations import L2
from .constants import OUTPUT_LAYER_NAMES, OUTPUT_LAYER_TAGS


def make_conv(out_ch, **kwargs):
    return Convolutional2D((3, 3), out_channels=out_ch, padding=1,
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


def make_down(out_chs, **kwargs):
    return Model(layers={
        'conv_block': make_conv_block(out_chs, **kwargs),
        'pool': MaxPool2D(2),
    }, relations={
        'conv_block': 0,
        'pool': 'conv_block',
        0: 'conv_block',
        1: 'pool',
    })


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

    ch_count_start = [in_channels, len(OUTPUT_LAYER_NAMES['monochrome'])]

    model = Model(layers={
        'start': make_conv_block(ch_count_start, True, **kwargs),
    }, relations={
        'start': 0,
        0: 'start',
    }, loss=SegmentationDice2D())

    return model


def make_unet(input_shape, optimizer=None):
    batch_size, height, width, in_channels = input_shape
    optimizer = Adam(lr=1e-2) if optimizer is None else optimizer

    kwargs = {
        'optimizer': optimizer,
        'trainable': True,
    }

    ch_count_downs = [[1], [2], [4]]
    ch_count_bottom = [8]
    ch_count_middles = [[1], [1], [1]]
    ch_count_ups = [[4], [8], [16]]
    ch_count_end = [4, 8, 16]

    assert len(ch_count_downs) == len(ch_count_middles) == len(ch_count_ups) > 0
    depth = len(ch_count_downs)

    assert OUTPUT_LAYER_TAGS[0] == 'monochrome'
    assert OUTPUT_LAYER_TAGS[1] == 'letter_spacing'
    assert OUTPUT_LAYER_TAGS[2] == 'paragraph'
    assert OUTPUT_LAYER_TAGS[3] == 'line'
    assert OUTPUT_LAYER_TAGS[4] == 'char_box'
    assert OUTPUT_LAYER_TAGS[5] == 'bit'

    non_end_output_layer_tags = ['monochrome']
    end_output_layer_tags = [
        tag for tag in OUTPUT_LAYER_TAGS
        if tag not in non_end_output_layer_tags
    ]

    ch_count_start = [in_channels, len(OUTPUT_LAYER_NAMES['monochrome'])]
    ch_count_ends = {
        'letter_spacing': [len(OUTPUT_LAYER_NAMES['letter_spacing'])],
        'paragraph': [len(OUTPUT_LAYER_NAMES['paragraph'])],
        'line': [len(OUTPUT_LAYER_NAMES['line'])],
        'char_box': [len(OUTPUT_LAYER_NAMES['char_box'])],
        'bit': [len(OUTPUT_LAYER_NAMES['bit'])],
    }

    model = Model(layers={
        'start': make_conv_block(ch_count_start, last_sigmoid=True, **kwargs),
        **{
            f'down_{i + 1}': make_down(ch_count_downs[i], **kwargs)
            for i in range(depth)
        },
        'bottom': make_conv_block(ch_count_bottom, **kwargs),
        **{
            f'middle_{i + 1}': make_conv_block(ch_count_middles[i], **kwargs)
            for i in range(depth)
        },
        **{
            f'up_{i + 1}': make_up(ch_count_ups[i], **kwargs)
            for i in range(depth)
        },
        'end': make_conv_block(ch_count_end, **kwargs),
        **{
            f'end_{tag}': make_conv_block(ch_count_ends[tag], last_sigmoid=True, **kwargs)
            for tag in end_output_layer_tags
        },
    }, relations={
        'start': 0,
        0: 'start',
        'down_1': 'start',
        **{f'down_{i + 2}': (f'down_{i + 1}', 1) for i in range(depth - 1)},
        'bottom': (f'down_{depth}', 1),
        **{f'middle_{i + 1}': (f'down_{i + 1}', 0) for i in range(depth)},
        f'up_{depth}': [f'middle_{depth}', 'bottom'],
        **{
            f'up_{i}': [f'middle_{i}', f'up_{i + 1}']
            for i in range(depth - 1, 0, -1)
        },
        'end': 'up_1',
        **{
            f'end_{tag}': 'end'
            for tag in end_output_layer_tags
        },
        **{
            i + len(OUTPUT_LAYER_TAGS) - len(end_output_layer_tags): f'end_{tag}'
            for i, tag in enumerate(end_output_layer_tags)
        },
    }, loss=SegmentationDice2D())

    return model
