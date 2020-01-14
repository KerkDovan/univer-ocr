import os
from datetime import datetime as dt

import numpy as np
from PIL import Image

from tqdm import tqdm

from ..interpreter import CropAndRotateLines, CropAndRotateParagraphs
from .constants import GENERATED_FILES_PATH, OUTPUT_LAYER_NAMES_PLAIN_IDS
from .datasets import train_dataset


def to_array(image):
    result = np.array(image)
    result = np.reshape(result, (1, *result.shape, 1))
    return result


def from_array(array, ch=0):
    result = array[0, :, :, ch]
    result = Image.fromarray(result)
    return result


def benchmark_one(dirpath, workers_count):
    total_paragraph_crop_time = dt.now() - dt.now()
    total_line_crop_time = dt.now() - dt.now()
    total_save_time = dt.now() - dt.now()

    crop_and_rotate_paragraphs = CropAndRotateParagraphs(workers_count)
    crop_and_rotate_lines = CropAndRotateLines(workers_count)
    print(f'Workers count: {crop_and_rotate_lines.workers_count}')

    for i in tqdm(range(len(train_dataset)), ascii=True):
        X, ys = train_dataset.get_images(i)
        monochrome = to_array(ys[OUTPUT_LAYER_NAMES_PLAIN_IDS['image_monochrome']])
        line_top = to_array(ys[OUTPUT_LAYER_NAMES_PLAIN_IDS['line_top']])
        line_center = to_array(ys[OUTPUT_LAYER_NAMES_PLAIN_IDS['line_center']])
        line_bottom = to_array(ys[OUTPUT_LAYER_NAMES_PLAIN_IDS['line_bottom']])
        line = np.concatenate([line_top, line_center, line_bottom], axis=3)
        mask = to_array(ys[OUTPUT_LAYER_NAMES_PLAIN_IDS['paragraph']])

        ts = dt.now()
        paragraphs = crop_and_rotate_paragraphs(
            mask, [monochrome, line])
        total_paragraph_crop_time += dt.now() - ts

        ts = dt.now()
        lines = crop_and_rotate_lines(paragraphs[1], [paragraphs[0]])[0]
        total_line_crop_time += dt.now() - ts

        for j in range(len(paragraphs[0])):
            ts = dt.now()

            cr_monochrome = from_array(paragraphs[0][j])
            cr_line_top = from_array(paragraphs[1][j], ch=0)
            cr_line_center = from_array(paragraphs[1][j], ch=1)
            cr_line_bottom = from_array(paragraphs[1][j], ch=2)

            cr_monochrome.save(dirpath / f'{i}_{j}_0_monochrome.png')
            cr_line_top.save(dirpath / f'{i}_{j}_1_line_top.png')
            cr_line_center.save(dirpath / f'{i}_{j}_2_line_center.png')
            cr_line_bottom.save(dirpath / f'{i}_{j}_3_line_bottom.png')

            for k in range(len(lines[j])):
                cr_line = from_array(lines[j][k])
                cr_line.save(dirpath / f'{i}_{j}_4_{k}_line.png')

            total_save_time += dt.now() - ts

    print(f'Total paragraph crop time: {total_paragraph_crop_time.total_seconds()} sec')
    print(f'Total line crop time: {total_line_crop_time.total_seconds()} sec')
    print(f'Total save time: {total_save_time.total_seconds()} sec')
    print(f'Timers:')
    print(f'  Crop and Rotate Paragraphs:')
    for k, v in crop_and_rotate_paragraphs.timers.items():
        print(f'    {k}: {v.total_seconds()} sec')
    print(f'  Crop and Rotate Lines:')
    for k, v in crop_and_rotate_lines.timers.items():
        print(f'    {k}: {v.total_seconds()} sec')
    print()


def main(*args, **kwargs):
    dirpath = GENERATED_FILES_PATH / 'crop_and_rotate_benchmark'
    dirpath.mkdir(parents=True, exist_ok=True)
    for fpath in dirpath.iterdir():
        os.remove(fpath)

    try:
        for workers_count in [1, 2, 4]:
            benchmark_one(dirpath, workers_count)

    except KeyboardInterrupt:
        print('Stopped by keyboard interrupt')


if __name__ == '__main__':
    main()
