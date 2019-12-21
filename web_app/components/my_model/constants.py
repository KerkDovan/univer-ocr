from pathlib import Path

from ..image_generator import LayeredImage
from ..primitives import BITS_COUNT

ALL_LAYER_NAMES = LayeredImage.layer_names
INPUT_LAYER_NAME = 'image'
OUTPUT_LAYER_TAGS = [
    'monochrome',
    'letter_spacing',
    'paragraph',
    'line',
    'char_box',
    'bit',
]
OUTPUT_LAYER_TAGS_IDS = {
    name: i for i, name in enumerate(OUTPUT_LAYER_TAGS)
}
OUTPUT_LAYER_NAMES = {
    OUTPUT_LAYER_TAGS[0]: ['image_monochrome'],
    OUTPUT_LAYER_TAGS[1]: ['letter_spacing'],
    OUTPUT_LAYER_TAGS[2]: ['paragraph'],
    OUTPUT_LAYER_TAGS[3]: ['line_top', 'line_center', 'line_bottom'],
    OUTPUT_LAYER_TAGS[4]: ['char_mask_box', 'char_full_box'],
    OUTPUT_LAYER_TAGS[5]: [f'bit_{i}' for i in range(BITS_COUNT)],
}
OUTPUT_LAYER_NAMES_PLAIN = [
    name
    for tag in OUTPUT_LAYER_TAGS
    for name in OUTPUT_LAYER_NAMES[tag]
]
OUTPUT_LAYER_NAMES_PLAIN_IDS = {
    name: i for i, name in enumerate(OUTPUT_LAYER_NAMES_PLAIN)
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
