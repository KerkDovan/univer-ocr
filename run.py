import importlib
import sys
import traceback


def main(module_name, use_gpu=False, *args, **kwargs):
    try:
        if module_name == 'train':
            import_path = module_name
        else:
            import_path = 'web_app.components.my_model.' + module_name
        imported = importlib.import_module(import_path)
        imported.main(use_gpu == 'True' or use_gpu is True, *args, **kwargs)

    except Exception:
        print(traceback.format_exc())


if __name__ == '__main__':
    main(*sys.argv[1:])
