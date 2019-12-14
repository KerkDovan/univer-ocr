from ..nn.layers import Concat, Convolutional2D, LeakyRelu, MaxPool2D, Sigmoid, Upsample2D
from ..nn.losses import SegmentationDice2D, SegmentationJaccard2D
from ..nn.models import Model
from ..nn.optimizers import Adam
from ..nn.regularizations import L2


def make_model(in_channels, out_channels, optimizer=None):
    optimizer = Adam(lr=0.05) if optimizer is None else optimizer

    def conv(out_channels):
        kernel = 3
        padding = (kernel - 1) // 2
        return Convolutional2D(
            (kernel, kernel), out_channels=out_channels, padding=padding, stride=1,
            regularizer=L2(0.01), optimizer=optimizer)

    def conv_relu(out_channels):
        return Model(layers={
            'conv': conv(out_channels),
            'leaky_relu': LeakyRelu(0.01),
        }, relations={
            'conv': 0,
            'leaky_relu': 'conv',
            0: 'leaky_relu',
        })

    model = Model(layers={
        'conv_start': conv_relu(in_channels),
        'conv_middle': conv_relu((in_channels + out_channels) // 2),
        'conv_end': conv(out_channels),
        'sigmoid': Sigmoid(),
    }, relations={
        'conv_start': 0,
        'conv_middle': 'conv_start',
        'conv_end': 'conv_middle',
        'sigmoid': 'conv_end',
        0: 'sigmoid',
    }, loss=SegmentationJaccard2D())

    return model


def make_unet(in_channels, out_channels, optimizer=None):
    optimizer = Adam(lr=1e-2) if optimizer is None else optimizer

    def conv(out_channels):
        return Convolutional2D((3, 3), out_channels=out_channels, padding=1,
                               regularizer=L2(0.01), optimizer=optimizer)

    def double_conv(out_channels, trainable):
        return Model(layers={
            'conv_1': conv(out_channels),
            'leaky_relu_1': LeakyRelu(0.01),
            'conv_2': conv(out_channels),
            'leaky_relu_2': LeakyRelu(0.01),
        }, relations={
            'conv_1': 0,
            'leaky_relu_1': 'conv_1',
            'conv_2': 'leaky_relu_1',
            'leaky_relu_2': 'conv_2',
            0: 'leaky_relu_2',
        }, trainable=trainable)

    def double_conv_end(out_channels, trainable):
        return Model(layers={
            'conv_1': conv(out_channels),
            'leaky_relu': LeakyRelu(0.01),
            'conv_2': conv(out_channels),
            'sigmoid': Sigmoid(),
        }, relations={
            'conv_1': 0,
            'leaky_relu': 'conv_1',
            'conv_2': 'leaky_relu',
            'sigmoid': 'conv_2',
            0: 'sigmoid',
        }, trainable=trainable)

    def down(out_channels, trainable):
        return Model(layers={
            'double_conv': double_conv(out_channels, trainable),
            'pool': MaxPool2D(2),
        }, relations={
            'double_conv': 0,
            'pool': 'double_conv',
            0: 'double_conv',
            1: 'pool',
        })

    def up(out_channels, trainable):
        return Model(layers={
            'upsample': Upsample2D(2),
            'concat': Concat(),
            'double_conv': double_conv(out_channels, trainable),
        }, relations={
            'upsample': 1,
            'concat': ['upsample', 0],
            'double_conv': 'concat',
            0: 'double_conv',
        })

    ch_count = [4, 8, 16, 32] + [64] + [32, 16, 8, 4]
    model = Model(layers={
        'down_1': down(ch_count[0], True),
        'down_2': down(ch_count[1], True),
        'down_3': down(ch_count[2], True),
        'down_4': down(ch_count[3], True),
        'middle': double_conv(ch_count[4], True),
        'up_1': up(ch_count[5], True),
        'up_2': up(ch_count[6], True),
        'up_3': up(ch_count[7], True),
        'up_4': up(ch_count[8], True),
        'end': double_conv_end(out_channels, True),
    }, relations={
        'down_1': 0,
        'down_2': ('down_1', 1),
        'down_3': ('down_2', 1),
        'down_4': ('down_3', 1),
        'middle': ('down_4', 1),
        'up_1': [('down_4', 0), 'middle'],
        'up_2': [('down_3', 0), 'up_1'],
        'up_3': [('down_2', 0), 'up_2'],
        'up_4': [('down_1', 0), 'up_3'],
        'end': 'up_4',
        0: 'end',
    }, loss=SegmentationDice2D())

    return model
