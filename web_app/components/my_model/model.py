from ..nn.layers import Concat, Convolutional2D, LeakyRelu, MaxPool2D, Relu, Sigmoid, Upsample2D
from ..nn.losses import SegmentationDice2D, SegmentationJaccard2D, SigmoidCrossEntropy
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
        'double_conv_down_1': double_conv(ch_count[0], True),
        'pool_1': MaxPool2D(2),
        'double_conv_down_2': double_conv(ch_count[1], True),
        'pool_2': MaxPool2D(2),
        'double_conv_down_3': double_conv(ch_count[2], True),
        'pool_3': MaxPool2D(2),
        'double_conv_down_4': double_conv(ch_count[3], True),
        'pool_4': MaxPool2D(2),
        'double_conv_mid': double_conv(ch_count[4], True),
        'up_1': up(ch_count[5], True),
        'up_2': up(ch_count[6], True),
        'up_3': up(ch_count[7], True),
        'up_4': up(ch_count[8], True),
        'double_conv_end': double_conv_end(out_channels, True),
    }, relations={
        'double_conv_down_1': 0,
        'pool_1': 'double_conv_down_1',
        'double_conv_down_2': 'pool_1',
        'pool_2': 'double_conv_down_2',
        'double_conv_down_3': 'pool_2',
        'pool_3': 'double_conv_down_3',
        'double_conv_down_4': 'pool_3',
        'pool_4': 'double_conv_down_4',
        'double_conv_mid': 'pool_4',
        'up_1': ['double_conv_down_4', 'double_conv_mid'],
        'up_2': ['double_conv_down_3', 'up_1'],
        'up_3': ['double_conv_down_2', 'up_2'],
        'up_4': ['double_conv_down_1', 'up_3'],
        'double_conv_end': 'up_4',
        0: 'double_conv_end',
    }, loss=SegmentationDice2D())

    return model
