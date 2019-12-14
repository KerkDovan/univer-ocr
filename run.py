import importlib
import sys
import traceback


def bool_convert(arg):
    return {'true': True, 'false': False}.get(str(arg).lower(), arg)


def main(module_name, use_gpu=False, *args, **kwargs):
    try:
        if module_name == 'train':
            import_path = module_name
        else:
            import_path = 'web_app.components.my_model.' + module_name
        imported = importlib.import_module(import_path)
        args = [bool_convert(arg) for arg in args]
        imported.main(use_gpu == 'True' or use_gpu is True, *args, **kwargs)

    except Exception as e:
        print(traceback.format_exc())
        raise e


if __name__ == '__main__':
    main(*sys.argv[1:])
