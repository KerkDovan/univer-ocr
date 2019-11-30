import importlib
import sys

import_path = 'web_app.components.nn.test.'


def main(test_name, use_gpu=False):
    imported = importlib.import_module(import_path + test_name)
    imported.main(use_gpu)


if __name__ == '__main__':
    main(*sys.argv[1:])
