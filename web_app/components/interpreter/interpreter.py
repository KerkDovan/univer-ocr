import os
from datetime import datetime as dt
from multiprocessing import Pool, Queue
from threading import Thread

import numpy as np
from scipy import ndimage

from ..primitives import BITS_COUNT, decode_char


def label_layer(layer):
    labels, cnt = ndimage.label(layer > np.mean(layer))
    result = []
    for l_id in range(cnt):
        result.append(labels == l_id + 1)
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


def get_center_of_mass(lines_top, lines_center, lines_bottom):
    top = [np.array(ndimage.center_of_mass(x)) for x in lines_top]
    center = [np.array(ndimage.center_of_mass(x)) for x in lines_center]
    bottom = [np.array(ndimage.center_of_mass(x)) for x in lines_bottom]
    return top, center, bottom


def rearrange_lines(lines_top, lines_center, lines_bottom):
    def cm(lines_top, lines_center, lines_bottom):
        cm_top, cm_center, cm_bottom = get_center_of_mass(
            lines_top, lines_center, lines_bottom)
        top = list(zip(cm_top, lines_top))
        center = list(zip(cm_center, lines_center))
        bottom = list(zip(cm_bottom, lines_bottom))
        return top, center, bottom
    top, center, bottom = cm(lines_top, lines_center, lines_bottom)
    lines_top = [
        sorted(top, key=lambda x: np.linalg.norm(с[0] - x[0]))[0][1]
        for с in center
    ]
    lines_bottom = [
        sorted(bottom, key=lambda x: np.linalg.norm(с[0] - x[0]))[0][1]
        for с in center
    ]

    _, h, w, _ = lines_top[0].shape
    dist_point = top[0][0] - bottom[0][0]
    while 0 < dist_point[1] < h or 0 < dist_point[2] < w:
        dist_point *= 1000

    if abs(dist_point[1]) > abs(dist_point[2]):
        if dist_point[1] < 0:
            def sort_key(x):
                return x[0][1]
            rotation = None
        elif dist_point[1] > h:
            def sort_key(x):
                return -x[0][1]
            rotation = 180
    else:
        if dist_point[2] < 0:
            def sort_key(x):
                return x[0][2]
            rotation = 270
        elif dist_point[2] > w:
            def sort_key(x):
                return -x[0][2]
            rotation = 90

    top, center, bottom = cm(lines_top, lines_center, lines_bottom)
    lines_top = [t[1] for t in sorted(top, key=sort_key)]
    lines_center = [c[1] for c in sorted(center, key=sort_key)]
    lines_bottom = [b[1] for b in sorted(bottom, key=sort_key)]
    return lines_top, lines_center, lines_bottom, rotation


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
    bits_layers = np.array([
        np.array(layers[f'bit_{i}']) > 0
        for i in range(BITS_COUNT)
    ]) & not_letter_spacing_layer
    pole = np.ones(bits_layers.shape[0], dtype=np.bool)

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
                bits_ids = np.argwhere(bits_layers[:, y, x] & pole)[:, 0]
                encoded = ''.join('1' if i in bits_ids else '0'
                                  for i in range(BITS_COUNT))
                decoded = decode_char(encoded)
                if decoded == 'unknown':
                    print(f'Could not recognize character at position [{x};{y}]')
                    continue
                res += decoded
            result[(p_id, l_id)] = ''.join(res)

    return result


def get_rotated(mask, arrays):
    def rotate(arr, angle):
        return ndimage.rotate(arr, angle, axes=(2, 1))

    def func(angle):
        rotated = rotate(mask, angle)
        _, region_y, region_x, _ = ndimage.find_objects(rotated)[0]
        return region_y.stop - region_y.start

    low, high = 0.0, 180.0
    while high - low > 1.0:
        a = low + (high - low) / 3
        b = high - (high - low) / 3
        if func(a) < func(b):
            high = b
        else:
            low = a

    angle = (high + low) / 2
    rotated_mask = rotate(mask, angle)
    _, region_y, region_x, _ = ndimage.find_objects(rotated_mask)[0]

    result = [
        rotate(arr, angle)[:, region_y, region_x, :]
        for arr in arrays
    ]

    return result


def crop_and_rotate_paragraph(mask, images):
    _, region_y, region_x, _ = ndimage.find_objects(mask)[0]
    cropped_mask = mask[:, region_y, region_x, :]
    cropped_images = [
        (image * mask)[:, region_y, region_x, :]
        for image in images
    ]
    result = get_rotated(cropped_mask, cropped_images)
    return result


class CropAndRotateParagraphs:
    def __init__(self, workers_count=None):
        self.input_queue = Queue()
        self.output_queue = Queue()
        self.workers_count = os.cpu_count() if workers_count is None else workers_count
        self.timers = {
            'label': dt.now() - dt.now(),
        }
        self.run_thread = Thread(target=self._run, daemon=True)
        self.run_thread.start()

    def __call__(self, masks, images):
        ts = dt.now()
        labeled_paragraph = label_layer(masks)
        self.timers['label'] += dt.now() - ts
        self.input_queue.put((masks, images, labeled_paragraph))
        result = self.output_queue.get()
        return result

    def _run(self):
        with Pool(self.workers_count) as pool:
            while True:
                masks, images, labeled_paragraph = self.input_queue.get()
                result = [[None for mask in labeled_paragraph] for image in images]
                async_res = []
                for paragraph_id, mask in enumerate(labeled_paragraph):
                    r = pool.apply_async(
                        crop_and_rotate_paragraph,
                        (mask, images))
                    async_res.append((paragraph_id, r))
                for paragraph_id, res in async_res:
                    res = res.get()
                    for image_id in range(len(images)):
                        result[image_id][paragraph_id] = res[image_id]
                self.output_queue.put(result)


def crop_and_rotate_line(top_mask, center_mask, bottom_mask, rotation, images):
    _, top_y, top_x, _ = ndimage.find_objects(top_mask)[0]
    _, center_y, center_x, _ = ndimage.find_objects(center_mask)[0]
    _, bottom_y, bottom_x, _ = ndimage.find_objects(bottom_mask)[0]
    y = slice(min(top_y.start, center_y.start, bottom_y.start),
              max(top_y.stop, center_y.stop, bottom_y.stop))
    x = slice(min(top_x.start, center_x.start, bottom_x.start),
              max(top_x.stop, center_x.stop, bottom_x.stop))
    result = []
    for i in range(len(images)):
        cropped = images[i][:, y, x, :]
        if rotation is not None:
            cropped = ndimage.rotate(cropped, rotation, axes=(2, 1))
        result.append(cropped)
    return result


class CropAndRotateLines:
    def __init__(self, workers_count=None):
        self.input_queue = Queue()
        self.output_queue = Queue()
        self.workers_count = os.cpu_count() if workers_count is None else workers_count
        self.timers = {
            'mask_mean': dt.now() - dt.now(),
            'rearrange': dt.now() - dt.now(),
        }
        self.run_thread = Thread(target=self._run, daemon=True)
        self.run_thread.start()

    def __call__(self, masks, arrays):
        self.input_queue.put((masks, arrays))
        result = self.output_queue.get()
        return result

    def _run(self):
        with Pool(self.workers_count) as pool:
            while True:
                masks, arrays = self.input_queue.get()

                result = [[] for array in arrays]
                async_res = []
                for paragraph_id, (mask, *subarrays) in enumerate(zip(masks, *arrays)):
                    for array_id in range(len(arrays)):
                        result[array_id].append([])

                    ts = dt.now()
                    top = mask[:, :, :, 0:1] > np.mean(mask[:, :, :, 0:1])
                    center = mask[:, :, :, 1:2] > np.mean(mask[:, :, :, 1:2])
                    bottom = mask[:, :, :, 2:3] > np.mean(mask[:, :, :, 2:3])
                    self.timers['mask_mean'] += dt.now() - ts

                    ts = dt.now()
                    top_mask, center_mask, bottom_mask, rotation = rearrange_lines(
                        label_layer(top), label_layer(center), label_layer(bottom))
                    self.timers['rearrange'] += dt.now() - ts

                    for line_id in range(len(top_mask)):
                        for array_id in range(len(arrays)):
                            result[array_id][paragraph_id].append(None)
                        index = (paragraph_id, line_id)
                        paragraph_data = (
                            top_mask[line_id], center_mask[line_id], bottom_mask[line_id],
                            rotation)
                        r = pool.apply_async(
                            crop_and_rotate_line,
                            (*paragraph_data, subarrays))
                        async_res.append((index, r))

                for (paragraph_id, line_id), res in async_res:
                    res = res.get()
                    for array_id in range(len(arrays)):
                        result[array_id][paragraph_id][line_id] = res[array_id]

                self.output_queue.put(result)
