import numpy as np
from scipy import ndimage

from ..primitives import CHARS, encode_char


def label_layer(layer):
    labels, cnt = ndimage.label(layer)
    result = np.zeros((cnt, *layer.shape), dtype=np.bool)
    for l_id in range(cnt):
        result[l_id] = labels == l_id + 1
    return result


def rearrange_points(points_top, points_center, points_bottom):
    new_top = [
        sorted(points_top, key=lambda x: np.linalg.norm(center - x))[0]
        for center in points_center
    ]
    new_bottom = [
        sorted(points_bottom, key=lambda x: np.linalg.norm(center - x))[0]
        for center in points_center
    ]
    return new_top, points_center, new_bottom


def get_sort_ids(center, vector, array):
    def pseudoscalar_prod(a, b):
        return a[1] * b[0] - b[1] * a[0]
    left = [(i, el) for i, el in enumerate(array) if pseudoscalar_prod(vector, el - center) <= 0]
    right = [(i, el) for i, el in enumerate(array) if pseudoscalar_prod(vector, el - center) > 0]
    left = sorted(left, key=lambda x: np.linalg.norm(x[1] - center), reverse=True)
    right = sorted(right, key=lambda x: np.linalg.norm(x[1] - center))
    return [i for i, _ in left + right]


def get_letter_sort_ids(cm_top, cm_bottom, letter_positions):
    return get_sort_ids(cm_bottom, cm_top - cm_bottom, letter_positions)


def get_line_sort_ids(cm_tops, cm_bottoms, cm_centers):
    def rotate90(vector):
        return np.array((vector[1], -vector[0]))
    return get_sort_ids(cm_bottoms[0], rotate90(cm_tops[0] - cm_bottoms[0]), cm_centers)


def iter_by_indices(iterable, indices):
    for index in indices:
        yield iterable[index]


def interpret(layers):
    paragraph_layer = np.array(layers['paragraph'])
    line_top_layer = np.array(layers['line_top'])
    line_center_layer = np.array(layers['line_center'])
    line_bottom_layer = np.array(layers['line_bottom'])
    not_letter_spacing_layer = ~(np.array(layers['letter_spacing']) > 0)
    char_full_box_layer = np.array(layers['char_full_box']) & not_letter_spacing_layer
    char_box_layers = np.array([
        np.array(layers['char_box_' + encode_char(char)]) > 0
        for char in CHARS
    ]) & not_letter_spacing_layer
    pole = np.ones(char_box_layers.shape[0], dtype=np.bool)

    char_box_objects = [
        ((y.start + y.stop - 1) // 2, (x.start + x.stop - 1) // 2)
        for y, x in ndimage.find_objects(ndimage.label(char_full_box_layer)[0])
    ]
    char_box_points = np.zeros_like(char_full_box_layer)
    for y, x in char_box_objects:
        char_box_points[y, x] = 1

    result = {}

    labeled_paragraph = label_layer(paragraph_layer)
    for p_id, paragraph_mask in enumerate(labeled_paragraph):
        p_y, p_x = ndimage.find_objects(paragraph_mask)[0]
        start = np.array([p_y.start, p_x.start])

        masked_line_top = label_layer(paragraph_mask[p_y, p_x] * line_top_layer[p_y, p_x])
        masked_line_center = label_layer(paragraph_mask[p_y, p_x] * line_center_layer[p_y, p_x])
        masked_line_bottom = label_layer(paragraph_mask[p_y, p_x] * line_bottom_layer[p_y, p_x])
        cm_top, cm_center, cm_bottom = rearrange_points(
            [np.array(ndimage.center_of_mass(x)) for x in masked_line_top],
            [np.array(ndimage.center_of_mass(x)) for x in masked_line_center],
            [np.array(ndimage.center_of_mass(x)) for x in masked_line_bottom])
        line_sort_ids = get_line_sort_ids(cm_top, cm_bottom, cm_center)

        for l_id, line in enumerate(iter_by_indices(masked_line_center, line_sort_ids)):
            s_y, s_x = ndimage.find_objects(line)[0]
            points = np.argwhere(
                line[s_y, s_x] * char_box_points[
                    start[0] + s_y.start:start[0] + s_y.stop,
                    start[1] + s_x.start:start[1] + s_x.stop])
            positions = [
                np.array((y + start[0] + s_y.start, x + start[1] + s_x.start))
                for y, x in points
            ]
            letter_sort_ids = get_letter_sort_ids(
                start + cm_top[l_id], start + cm_bottom[l_id], positions)
            res = ''
            for y, x in iter_by_indices(positions, letter_sort_ids):
                possible_char_ids = np.argwhere(char_box_layers[:, y, x] & pole)
                if possible_char_ids.shape[0] == 0:
                    print(f'Cannot recognize a char at position [{x};{y}]')
                    continue
                char_id = possible_char_ids[0, 0]
                res += CHARS[char_id]
            result[(p_id, l_id)] = ''.join(res)

    return result
