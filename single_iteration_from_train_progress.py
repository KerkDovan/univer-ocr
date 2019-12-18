import os
import shutil
import sys

from web_app.components.my_model.constants import (
    SINGLE_ITERATION_FROM_TRAIN_PROGRESS_PATH, TRAIN_PROGRESS_PATH)


def main(epoch_id, train_val='train', iter_id=0):
    epoch_id = int(epoch_id)
    assert train_val in ['train', 'validation']
    iter_id = int(iter_id)

    if SINGLE_ITERATION_FROM_TRAIN_PROGRESS_PATH.exists():
        for fpath in SINGLE_ITERATION_FROM_TRAIN_PROGRESS_PATH.iterdir():
            os.remove(fpath)
    else:
        os.mkdir(SINGLE_ITERATION_FROM_TRAIN_PROGRESS_PATH)

    for picture_type in TRAIN_PROGRESS_PATH.iterdir():
        for i, pic in enumerate(['X', 'y', 'pred', 'thresholded']):
            pic_path = picture_type / (
                f'{epoch_id}_{train_val}_{iter_id}_{i + 1}_{pic}.png')
            new_path = SINGLE_ITERATION_FROM_TRAIN_PROGRESS_PATH / (
                f'{epoch_id}_{train_val}_{iter_id}_{picture_type.name}_{i + 1}_{pic}.png')

            shutil.copyfile(pic_path, new_path)


if __name__ == '__main__':
    main(*sys.argv[1:])
