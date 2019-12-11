from tqdm import tqdm

from .constants import TRAIN_DATA_PATH, VALIDATION_DATA_PATH
from .train_data_generator import DataGenerator, generate_picture


def main(*args, **kwargs):
    data_generator = DataGenerator(640, 480, 4, generator_func=generate_picture)
    data_generator.start()

    train_path = TRAIN_DATA_PATH
    val_path = VALIDATION_DATA_PATH

    train_path.mkdir(parents=True, exist_ok=True)
    val_path.mkdir(parents=True, exist_ok=True)

    for i in tqdm(range(10000)):
        images = data_generator.get_data()
        for layer_name, image in images.items():
            image.save(train_path / f'{i}_{layer_name}.png')

    for i in tqdm(range(1000)):
        images = data_generator.get_data()
        for layer_name, image in images.items():
            image.save(val_path / f'{i}_{layer_name}.png')


if __name__ == '__main__':
    main()
