import importlib
import sys

import_path = 'web_app.components.nn.test.'


def main(test_name):
    imported = importlib.import_module(import_path + test_name)
    imported.main()


if __name__ == '__main__':
    main(sys.argv[1])
