from tqdm import tqdm

from .constants import (
    TRAIN_DATA_PATH, TRAIN_DATASET_LENGTH, VALIDATION_DATA_PATH, VALIDATION_DATASET_LENGTH)
from .train_data_generator import DataGenerator, generate_picture


def main(*args, **kwargs):
    data_generator = DataGenerator(
        generator_func=generate_picture, func_args=(640, 480, False))
    data_generator.start()

    train_path = TRAIN_DATA_PATH
    val_path = VALIDATION_DATA_PATH

    train_path.mkdir(parents=True, exist_ok=True)
    val_path.mkdir(parents=True, exist_ok=True)

    for i in tqdm(range(TRAIN_DATASET_LENGTH)):
        images = data_generator.get_data()
        for layer_name, image in images.items():
            image.save(train_path / f'{i}_{layer_name}.png')

    for i in tqdm(range(VALIDATION_DATASET_LENGTH)):
        images = data_generator.get_data()
        for layer_name, image in images.items():
            image.save(val_path / f'{i}_{layer_name}.png')

    data_generator.stop()


if __name__ == '__main__':
    main()
