from pathlib import Path

from ..image_generator import LayeredImage
from ..primitives import BITS_COUNT

ALL_LAYER_NAMES = LayeredImage.layer_names
LAYER_TAGS = [
    'image',
    'monochrome',
    'paragraph',
    'line',
    'char',
]
LAYER_TAGS_IDS = {
    name: i for i, name in enumerate(LAYER_TAGS)
}
LAYER_NAMES = {
    LAYER_TAGS[0]: ['image'],
    LAYER_TAGS[1]: ['image_monochrome'],
    LAYER_TAGS[2]: ['paragraph'],
    LAYER_TAGS[3]: ['line_top', 'line_center', 'line_bottom'],
    LAYER_TAGS[4]: [
        *[f'bit_{i}' for i in range(BITS_COUNT)],
        'letter_spacing',
    ]
}
LAYER_NAMES_PLAIN = [
    name
    for tag in LAYER_TAGS
    for name in LAYER_NAMES[tag]
]
LAYER_NAMES_PLAIN_IDS = {
    name: i for i, name in enumerate(LAYER_NAMES_PLAIN)
}

MODEL_WEIGHTS_FILE_PATH = Path('web_app', 'components', 'my_model', 'model_weights.json')
GENERATED_FILES_PATH = Path('generated_files')
TRAIN_DATA_PATH = (
    GENERATED_FILES_PATH / 'data' / 'train')
VALIDATION_DATA_PATH = (
    GENERATED_FILES_PATH / 'data' / 'validation')
TRAIN_PROGRESS_PATH = (
    GENERATED_FILES_PATH / 'train_progress')
SINGLE_ITERATION_FROM_TRAIN_PROGRESS_PATH = (
    GENERATED_FILES_PATH / 'single_iteration_from_train_progress')
PREDICTION_RESULT_PATH = (
    GENERATED_FILES_PATH / 'prediction_result')
LAYERS_OUTPUTS_PATH = (
    GENERATED_FILES_PATH / 'layers_outputs')

TRAIN_DATASET_LENGTH = 100
VALIDATION_DATASET_LENGTH = 10
