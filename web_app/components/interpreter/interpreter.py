import os
import signal
from datetime import datetime as dt
from queue import Empty
from threading import Thread
from time import sleep

import numpy as np
from scipy import ndimage

from ..primitives import BITS_COUNT, decode_char
from .parallelism import ERRORS_TO_STOP, MP


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


def put_to_queue(queue, data):
    try:
        queue.put(data)
    except ERRORS_TO_STOP:
        return


def get_from_queue(queue):
    try:
        return queue.get()
    except ERRORS_TO_STOP:
        exit(0)


def rotate_array(array, angle=None):
    if angle is None:
        return array
    return ndimage.rotate(array, angle, axes=(2, 1))


class FindObjectHeightInRotated:
    def __init__(self, manager, done):
        self.manager = manager
        self.input_queue = self.manager.Queue()
        self.output_queue = self.manager.Queue()
        self.done = done
        self.worker = MP.Process(
            target=self._run, daemon=True,
            args=(self.done, self._func, self.input_queue, self.output_queue))

    def start(self):
        self.worker.start()

    def stop(self):
        self.done.set()
        sleep(0.001)

    def __del__(self):
        self.stop()

    @staticmethod
    def _run(done, func, input_queue, output_queue):
        while not done.is_set():
            try:
                args = input_queue.get(True, 0.001)
                result = func(*args)
                put_to_queue(output_queue, result)
            except Empty:
                continue
            except ERRORS_TO_STOP:
                break

    @staticmethod
    def _func(array, angle):
        rotated = rotate_array(array, angle)
        _, region_y, _, _ = ndimage.find_objects(rotated)[0]
        return region_y.stop - region_y.start


class CropAndRotateSingleParagraph:
    def __init__(self, manager, workers_count=None, find_rotation=True, EPS=1.0):
        self.manager = manager
        self.input_queue = self.manager.Queue()
        self.output_queue = self.manager.Queue()
        self.workers_count = os.cpu_count() if workers_count is None else workers_count
        self.find_rotation = find_rotation
        self.EPS = EPS
        self.done = MP.mp.Event()
        self.height_finders = [
            FindObjectHeightInRotated(self.manager, self.done)
            for _ in range(2 * self.workers_count)
        ] if self.find_rotation else []
        queues = [
            (
                self.height_finders[2 * i].input_queue,
                self.height_finders[2 * i].output_queue,
                self.height_finders[2 * i + 1].input_queue,
                self.height_finders[2 * i + 1].output_queue
            ) if self.find_rotation else (None, None, None, None)
            for i in range(self.workers_count)
        ]
        self.workers = [
            MP.Process(target=self._run, daemon=True, args=(
                self.done, self._func, self.find_rotation, self.EPS,
                self.input_queue, self.output_queue, *queues[i]))
            for i in range(self.workers_count)
        ]

    def start(self):
        for worker in self.workers:
            worker.start()
        if self.find_rotation:
            for hf in self.height_finders:
                hf.start()

    def stop(self):
        self.done.set()
        sleep(0.001)

    def __del__(self):
        self.stop()

    def put(self, label, mask, arrays):
        put_to_queue(self.input_queue, (label, mask, arrays))

    def map(self, paragraphs):
        assert isinstance(paragraphs, dict)
        for label, (mask, arrays) in paragraphs.items():
            self.put(label, mask, arrays)
        result = {}
        counter = 0
        while not self.done.is_set() and counter < len(paragraphs):
            try:
                label, res = self.output_queue.get(True, 0.001)
            except Empty:
                continue
            except ERRORS_TO_STOP:
                break
            result[label] = res
            counter += 1
        return result

    @staticmethod
    def _run(done, func, find_rotation, EPS, input_queue, output_queue,
             a_in_queue, a_out_queue, b_in_queue, b_out_queue):
        while not done.is_set():
            try:
                label, mask, arrays = input_queue.get(True, 0.001)
                _, region_y, region_x, _ = ndimage.find_objects(mask)[0]
                cropped_mask = mask[:, region_y, region_x, :]
                cropped_arrays = [
                    (image * mask)[:, region_y, region_x, :]
                    for image in arrays
                ]
                result = func(
                    find_rotation, EPS,
                    a_in_queue, a_out_queue, b_in_queue, b_out_queue,
                    cropped_mask, cropped_arrays)
                put_to_queue(output_queue, (label, result))
            except Empty:
                continue
            except ERRORS_TO_STOP:
                break

    @staticmethod
    def _func(find_rotation, EPS, a_in_queue, a_out_queue, b_in_queue, b_out_queue, mask, arrays):
        if find_rotation:
            low, high = 0.0, 180.0
            while high - low > EPS:
                a = low + (high - low) / 3
                b = high - (high - low) / 3
                put_to_queue(a_in_queue, (mask, a))
                put_to_queue(b_in_queue, (mask, b))
                height_a = get_from_queue(a_out_queue)
                height_b = get_from_queue(b_out_queue)
                if height_a < height_b:
                    high = b
                else:
                    low = a
            angle = (high + low) / 2
            if angle < EPS:
                angle = None
        else:
            angle = None

        rotated_mask = rotate_array(mask, angle)
        _, region_y, region_x, _ = ndimage.find_objects(rotated_mask)[0]

        result = [
            rotate_array(arr, angle)[:, region_y, region_x, :]
            for arr in arrays
        ]
        return result


class CropAndRotateParagraphs:
    def __init__(self, workers_count=None, find_rotation=True):
        self.manager = MP.mp.Manager()
        self.timers = {
            'label': dt.now() - dt.now(),
        }
        self.carsp = CropAndRotateSingleParagraph(self.manager, workers_count, find_rotation)
        self.carsp.start()

    def __del__(self):
        self.carsp.stop()

    def __call__(self, masks, images):
        ts = dt.now()
        labeled_paragraph = label_layer(masks)
        self.timers['label'] += dt.now() - ts

        paragraphs = {
            paragraph_id: (mask, images)
            for paragraph_id, mask in enumerate(labeled_paragraph)
        }
        rotated = self.carsp.map(paragraphs)

        result = [[None for mask in labeled_paragraph] for image in images]
        for paragraph_id, res in rotated.items():
            for image_id in range(len(images)):
                result[image_id][paragraph_id] = res[image_id]

        return result


class CropRotateAndZoomLines:
    def __init__(self, workers_count=None, zoomed_height=None, minimal_width=None):
        self.manager = MP.mp.Manager()
        self.input_queue = self.manager.Queue()
        self.output_queue = self.manager.Queue()
        self.workers_count = os.cpu_count() if workers_count is None else workers_count
        self.zoomed_height = zoomed_height
        self.minimal_width = minimal_width
        self.done = MP.mp.Event()
        self.timers = {
            'mask_mean': dt.now() - dt.now(),
            'rearrange': dt.now() - dt.now(),
            'slices': dt.now() - dt.now(),
            'crop_and_rotate': dt.now() - dt.now(),
        }
        self.run_thread = Thread(target=self._run, daemon=True)
        self.run_thread.start()

    def __del__(self):
        self.done.set()
        sleep(0.001)

    def __call__(self, masks, arrays):
        put_to_queue(self.input_queue, (masks, arrays))
        result = get_from_queue(self.output_queue)
        return result

    @staticmethod
    def init_worker():
        if MP.is_multiprocessing_used:
            signal.signal(signal.SIGINT, signal.SIG_IGN)

    def _run(self):
        def thresholded(arr):
            return arr > 0.5 * (np.mean(arr) + np.max(arr))

        with MP.Pool(self.workers_count, self.init_worker) as pool:
            while not self.done.is_set():
                try:
                    masks, arrays = self.input_queue.get(True, 0.001)
                except Empty:
                    continue
                except ERRORS_TO_STOP:
                    break

                rearrange_ts = dt.now()

                async_rearranged = []
                for mask, *subarrays in zip(masks, *arrays):
                    mask_mean_ts = dt.now()
                    try:
                        top = thresholded(mask[:, :, :, 0:1])
                        center = thresholded(mask[:, :, :, 1:2])
                        bottom = thresholded(mask[:, :, :, 2:3])
                    except TypeError:
                        exit(0)
                    self.timers['mask_mean'] += dt.now() - mask_mean_ts

                    r = pool.apply_async(rearrange_lines, (
                        label_layer(top), label_layer(center), label_layer(bottom)))
                    async_rearranged.append(r)

                slices_ts = dt.now()

                async_slices = []
                result = [[] for array in arrays]
                for paragraph_id, _ in enumerate(zip(*arrays)):
                    for array_id in range(len(arrays)):
                        result[array_id].append([])
                    top_mask, center_mask, bottom_mask, rotation = (
                        async_rearranged[paragraph_id].get())
                    for line_id in range(len(top_mask)):
                        for array_id in range(len(arrays)):
                            result[array_id][paragraph_id].append(None)
                        index = (paragraph_id, line_id)
                        r = pool.apply_async(self._func1, (
                            top_mask[line_id], center_mask[line_id], bottom_mask[line_id]))
                        async_slices.append((index, r, rotation))

                self.timers['rearrange'] += dt.now() - rearrange_ts
                crop_and_rotate_ts = dt.now()

                async_res = []
                for (paragraph_id, line_id), slices, rotation in async_slices:
                    y, x = slices.get()
                    for array_id in range(len(arrays)):
                        index = (array_id, paragraph_id, line_id)
                        r = pool.apply_async(self._func2, (
                            arrays[array_id][paragraph_id], y, x, rotation,
                            self.zoomed_height, self.minimal_width))
                        async_res.append((index, r))

                self.timers['slices'] += dt.now() - slices_ts

                for (array_id, paragraph_id, line_id), res in async_res:
                    result[array_id][paragraph_id][line_id] = res.get()
                self.timers['crop_and_rotate'] += dt.now() - crop_and_rotate_ts

                put_to_queue(self.output_queue, result)

    @staticmethod
    def _func1(top_mask, center_mask, bottom_mask):
        _, top_y, top_x, _ = ndimage.find_objects(top_mask)[0]
        _, center_y, center_x, _ = ndimage.find_objects(center_mask)[0]
        _, bottom_y, bottom_x, _ = ndimage.find_objects(bottom_mask)[0]
        y = slice(min(top_y.start, center_y.start, bottom_y.start),
                  max(top_y.stop, center_y.stop, bottom_y.stop))
        x = slice(min(top_x.start, center_x.start, bottom_x.start),
                  max(top_x.stop, center_x.stop, bottom_x.stop))
        return y, x

    @staticmethod
    def _func2(image, y, x, rotation, zoomed_height, minimal_width):
        final_image = image[:, y, x, :]

        if rotation is not None:
            final_image = rotate_array(final_image, rotation)

        if zoomed_height is not None:
            height = final_image.shape[1]
            zf = zoomed_height / height
            final_image = ndimage.zoom(final_image, (1, zf, zf, 1))

        if minimal_width is not None and final_image.shape[2] < minimal_width:
            bs, h, w, ch = final_image.shape
            shape = (bs, h, minimal_width, ch)
            tmp = np.zeros(shape, dtype=final_image.dtype)
            tmp[:, :, :w, :] = final_image
            final_image = tmp

        return final_image
