from ..image_generator import LayeredImage

ALL_LAYER_NAMES = LayeredImage.layer_names
INPUT_LAYER_NAME = 'image'
OUTPUT_LAYER_NAMES = [name for name in ALL_LAYER_NAMES if name != INPUT_LAYER_NAME]
