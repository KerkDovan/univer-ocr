import numpy as np
from scipy import ndimage

from ..primitives import CHARS, encode_char


def label_layer(layer):
    labels, cnt = ndimage.label(layer)
    result = np.zeros((cnt, *layer.shape), dtype=np.bool)
    for l_id in range(cnt):
        result[l_id] = labels == l_id + 1
    return result


def rearrange_lines(masked_line_top, masked_line_center, masked_line_bottom):
    for i, center in enumerate(masked_line_center):
        top = center * masked_line_top
        j = np.argmax(np.count_nonzero(top, axis=(1, 2)))
        if i != j:
            masked_line_top[[i, j]] = masked_line_top[[j, i]]

        bottom = center * masked_line_bottom
        j = np.argmax(np.count_nonzero(bottom, axis=(1, 2)))
        if i != j:
            masked_line_bottom[[i, j]] = masked_line_bottom[[j, i]]

    return masked_line_top, masked_line_center, masked_line_bottom


def group_lines(line_top, line_center, line_bottom):
    result = np.zeros_like(line_center, dtype=np.bool)
    for i in range(line_center.shape[0]):
        result[i] = (line_top[i] + line_center[i] + line_bottom[i]) > 0
    return result


def interpret(layers):
    paragraph_layer = np.array(layers['paragraph'])
    line_top_layer = np.array(layers['line_top'])
    line_center_layer = np.array(layers['line_center'])
    line_bottom_layer = np.array(layers['line_bottom'])
    not_letter_spacing_layer = ~(np.array(layers['letter_spacing']) > 0)
    char_box_layers = np.array([
        np.array(layers['char_box_' + encode_char(char)]) > 0
        for char in CHARS
    ]) & not_letter_spacing_layer

    char_box_objects = {}
    for i, layer in enumerate(char_box_layers):
        labels, _ = ndimage.label(layer)
        char_box_objects[CHARS[i]] = [
            ((y.start + y.stop - 1) // 2, (x.start + x.stop - 1) // 2)
            for y, x in ndimage.find_objects(labels)
        ]

    char_box_points = np.zeros_like(char_box_layers)
    for i, char in enumerate(CHARS):
        for y, x in char_box_objects[char]:
            char_box_points[i, y, x] = 1

    result = {}

    labeled_paragraph = label_layer(paragraph_layer)
    for p_id, paragraph_mask in enumerate(labeled_paragraph):
        masked_line_top, masked_line_center, masked_line_bottom = rearrange_lines(
            label_layer(paragraph_mask * line_top_layer),
            label_layer(paragraph_mask * line_center_layer),
            label_layer(paragraph_mask * line_bottom_layer))
        lines = group_lines(masked_line_top, masked_line_center, masked_line_bottom)
        for l_id, line in enumerate(lines):
            s_y, s_x = ndimage.find_objects(line)[0]
            points = np.argwhere(line[s_y, s_x] * char_box_points[:, s_y, s_x])
            tmp = [
                ((s_x.start + x, s_y.start + y), CHARS[i]) for i, y, x in points
            ]
            sorted_tmp = [x[1] for x in sorted(tmp)]
            result[f'{p_id}_{l_id}'] = ''.join(sorted_tmp)

    return result
