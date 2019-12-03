import importlib
import sys
import traceback

import_path = 'web_app.components.nn.test.'


def main(test_name, use_gpu=False):
    try:
        imported = importlib.import_module(import_path + test_name)
        imported.main(use_gpu == 'True' or use_gpu is True)

    except Exception:
        print(traceback.format_exc())


if __name__ == '__main__':
    main(*sys.argv[1:])
