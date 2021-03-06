import gc
import os
import time
from datetime import datetime as dt

import numpy as np
from PIL import Image

from tqdm import tqdm

from ..interpreter import (
    MP, CropAndRotateParagraphs, CropRotateAndZoomLines, LabelChar, PredToText)
from .constants import GENERATED_FILES_PATH
from .datasets import train_dataset


def to_array(image):
    result = np.array(image)
    result = np.reshape(result, (1, *result.shape, 1)) / 255
    return result


def from_array(array, ch=0):
    result = (array[0, :, :, ch] * 255).astype(np.uint8)
    result = Image.fromarray(result)
    return result


def benchmark_one(dirpath, workers_count):
    total_paragraph_crop_time = dt.now() - dt.now()
    total_line_crop_time = dt.now() - dt.now()
    total_char_label_time = dt.now() - dt.now()
    total_pred_to_text_time = dt.now() - dt.now()
    total_save_time = dt.now() - dt.now()

    crop_and_rotate_paragraphs = CropAndRotateParagraphs(workers_count, True)
    crop_rotate_and_zoom_lines = CropRotateAndZoomLines(workers_count, 32, 200)
    label_char = LabelChar(workers_count)
    pred_to_text = PredToText(workers_count)
    print(f'Workers count: {workers_count}')

    for i in tqdm(range(len(train_dataset)), ascii=True):
        layers = train_dataset.get_images(i)
        monochrome = to_array(layers['image_monochrome'])
        line_top = to_array(layers['line_top'])
        line_bottom = to_array(layers['line_bottom'])
        line = np.concatenate([line_top, line_bottom], axis=3)
        bits_and_letter_spacings = np.concatenate([
            *[to_array(layers[f'bit_{i}']) for i in range(8)],
            to_array(layers['letter_spacing'])
        ], axis=3)
        mask = to_array(layers['paragraph'])

        ts = dt.now()
        paragraphs = crop_and_rotate_paragraphs(
            mask, [monochrome, line, bits_and_letter_spacings])
        total_paragraph_crop_time += dt.now() - ts

        ts = dt.now()
        lines, bit_lines = crop_rotate_and_zoom_lines(
            paragraphs[1], [paragraphs[0], paragraphs[2]])
        total_line_crop_time += dt.now() - ts

        ts = dt.now()
        labels = label_char(bit_lines)
        total_char_label_time += dt.now() - ts

        ts = dt.now()
        text = pred_to_text(labels)
        total_pred_to_text_time += dt.now() - ts

        for j in range(len(paragraphs[0])):
            ts = dt.now()

            cr_monochrome = from_array(paragraphs[0][j])
            cr_line_top = from_array(paragraphs[1][j], ch=0)
            cr_line_bottom = from_array(paragraphs[1][j], ch=1)
            cr_bits = [from_array(paragraphs[2][j], ch=i) for i in range(8)]
            cr_letter_spacing = from_array(paragraphs[2][j], ch=8)

            cr_monochrome.save(dirpath / f'{i}_{j}_0_monochrome.png')
            cr_line_top.save(dirpath / f'{i}_{j}_1_line_top.png')
            cr_line_bottom.save(dirpath / f'{i}_{j}_2_line_bottom.png')
            for k, bit in enumerate(cr_bits):
                bit.save(dirpath / f'{i}_{j}_3_bit_{k}.png')
            cr_letter_spacing.save(dirpath / f'{i}_{j}_4_letter_spacing.png')

            text_file = open(dirpath / f'{i}_{j}_0_text.txt', 'w', encoding='utf-8')
            for k in range(len(lines[j])):
                concatenated = np.concatenate([
                    lines[j][k],
                    *[bit_lines[j][k][:, :, :, ch:ch + 1] for ch in range(8)],
                    bit_lines[j][k][:, :, :, 8:9],
                ], axis=1)
                cr_line = from_array(concatenated)
                cr_line.save(dirpath / f'{i}_{j}_5_{k}_line.png')
                print(text[j][k], file=text_file)
            text_file.close()

            total_save_time += dt.now() - ts

    print(f'Total paragraph crop time: {total_paragraph_crop_time.total_seconds()} sec')
    print(f'Total line crop time: {total_line_crop_time.total_seconds()} sec')
    print(f'Total char label time: {total_char_label_time.total_seconds()} sec')
    print(f'Total save time: {total_save_time.total_seconds()} sec')
    print(f'Timers:')
    print(f'  Crop and Rotate Paragraphs:')
    for k, v in crop_and_rotate_paragraphs.timers.items():
        print(f'    {k}: {v.total_seconds()} sec')
    print(f'  Crop and Rotate Lines:')
    for k, v in crop_rotate_and_zoom_lines.timers.items():
        print(f'    {k}: {v.total_seconds()} sec')
    print()


def main(*args, **kwargs):
    dirpath = GENERATED_FILES_PATH / 'crop_and_rotate_benchmark'
    dirpath.mkdir(parents=True, exist_ok=True)
    for fpath in dirpath.iterdir():
        os.remove(fpath)

    print(f'os.cpu_count() for this machine is {os.cpu_count()}\n')

    try:
        workers_counts = [1, 2, 4]

        print('Using threading')
        MP.use_threading()
        for count in workers_counts:
            benchmark_one(dirpath, count)
            time.sleep(1)
            gc.collect()

        print('Using multiprocessing')
        MP.use_multiprocessing()
        for count in workers_counts:
            benchmark_one(dirpath, count)
            time.sleep(1)
            gc.collect()

    except KeyboardInterrupt:
        print('Stopped by keyboard interrupt')


if __name__ == '__main__':
    main()
