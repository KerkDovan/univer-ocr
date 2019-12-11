from pathlib import Path

from ..image_generator import LayeredImage

ALL_LAYER_NAMES = LayeredImage.layer_names
INPUT_LAYER_NAME = 'image'
OUTPUT_LAYER_NAMES = [name for name in ALL_LAYER_NAMES if name != INPUT_LAYER_NAME]

MODEL_WEIGHTS_FILE_PATH = Path('web_app', 'components', 'my_model', 'model_weights.json')
GENERATED_FILES_PATH = Path('generated_files')
TRAIN_DATA_PATH = GENERATED_FILES_PATH / 'data' / 'train'
VALIDATION_DATA_PATH = GENERATED_FILES_PATH / 'data' / 'validation'
TRAIN_PROGRESS_PATH = GENERATED_FILES_PATH / 'train_progress'
PREDICTION_RESULT_PATH = GENERATED_FILES_PATH / 'prediction_result'
