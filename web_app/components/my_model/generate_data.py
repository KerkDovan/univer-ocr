from pathlib import Path

from tqdm import tqdm

from .train_data import DataGenerator, generate_picture


def main(*args, **kwargs):
    data_generator = DataGenerator(640, 480, 4, generator_func=generate_picture)
    data_generator.start()

    path = Path('web_app', 'components', 'my_model', 'data')
    train_path = path / 'train'
    val_path = path / 'validation'

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
