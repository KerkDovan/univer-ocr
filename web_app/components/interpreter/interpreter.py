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


def sort_letters(cm_top, cm_bottom, letter_positions):
    def angle(coords):
        a = np.angle(coords[1] - coords[0] * 1j, True)
        return a + 2 * np.angle(-1, True) if a < 0 else a
    center_vec = (cm_top - cm_bottom) * 10
    center = cm_top + center_vec
    cv_angle = angle(center_vec)
    angles = [(angle(pos - center), pos) for pos in letter_positions]
    angles = [(a + cv_angle if a < cv_angle else a, p) for a, p in angles]
    return [x[1] for x in sorted(angles)]


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

        for l_id, line in enumerate(masked_line_center):
            s_y, s_x = ndimage.find_objects(line)[0]
            points = np.argwhere(
                line[s_y, s_x] * char_box_points[
                    start[0] + s_y.start:start[0] + s_y.stop,
                    start[1] + s_x.start:start[1] + s_x.stop])
            positions = [
                np.array((y + start[0] + s_y.start, x + start[1] + s_x.start))
                for y, x in points
            ]
            positions = sort_letters(start + cm_top[l_id], start + cm_bottom[l_id], positions)
            res = ''
            for y, x in positions:
                possible_char_ids = np.argwhere(char_box_layers[:, y, x] & pole)
                if possible_char_ids.shape[0] == 0:
                    print(f'Cannot recognize a char at position [{x};{y}]')
                    continue
                char_id = possible_char_ids[0, 0]
                res += CHARS[char_id]
            result[(p_id, l_id)] = ''.join(res)

    return result
