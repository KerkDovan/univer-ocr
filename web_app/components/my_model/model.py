from ..nn.layers import Concat, Convolutional2D, MaxPool2D, Relu, Upsample2D
from ..nn.losses import SegmentationDice2D
from ..nn.models import Model
from ..nn.regularizations import L2


def make_unet(out_channels):
    def double_conv(out_channels):
        return Model(layers={
            'conv_1': Convolutional2D((3, 3), out_channels=out_channels, padding=1,
                                      regularizer=L2(0.1)),
            'relu_1': Relu(),
            'conv_2': Convolutional2D((3, 3), out_channels=out_channels, padding=1,
                                      regularizer=L2(0.1)),
            'relu_2': Relu(),
        }, relations={
            'conv_1': 0,
            'relu_1': 'conv_1',
            'conv_2': 'relu_1',
            'relu_2': 'conv_2',
            0: 'relu_2',
        })

    def up(out_channels):
        return Model(layers={
            'upsample': Upsample2D(2),
            'concat': Concat(),
            'double_conv': double_conv(out_channels),
        }, relations={
            'upsample': 1,
            'concat': ['upsample', 0],
            'double_conv': 'concat',
            0: 'double_conv',
        })

    ch_count = [4, 8, 12, 24] + [32] + [48, 64, 96, 128]
    model = Model(layers={
        'double_conv_down_1': double_conv(ch_count[0]),
        'pool_1': MaxPool2D(2),
        'double_conv_down_2': double_conv(ch_count[1]),
        'pool_2': MaxPool2D(2),
        'double_conv_down_3': double_conv(ch_count[2]),
        'pool_3': MaxPool2D(2),
        'double_conv_down_4': double_conv(ch_count[3]),
        'pool_4': MaxPool2D(2),
        'double_conv_mid': double_conv(ch_count[4]),
        'up_1': up(ch_count[5]),
        'up_2': up(ch_count[6]),
        'up_3': up(ch_count[7]),
        'up_4': up(ch_count[8]),
        'double_conv_end': double_conv(out_channels),
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
