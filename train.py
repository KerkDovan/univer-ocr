import sys
import traceback

from socketIO_client import BaseNamespace, SocketIO

from web_app.components.my_model.train import init_emitter, train_model


def main(use_gpu=False, console_mode=True, show_progress_bar=False, save_train_progress=False):
    client = None

    if console_mode:
        print('Running in console mode')

    else:
        try:
            socketIO = SocketIO('127.0.0.1', 80, Namespace=BaseNamespace)
            client = socketIO.define(BaseNamespace, '/train-ws')
            init_emitter(client)

        except Exception:
            print(f'Cannot connect to socket.io server, running in console mode')

    try:
        train_model(
            use_gpu == 'True' or use_gpu is True,
            show_progress_bar == 'True' or show_progress_bar is True,
            save_train_progress == 'True' or save_train_progress is True,
        )

    except KeyboardInterrupt:
        print(f'Stopped by keyboard interrupt')

    except Exception as e:
        print(traceback.format_exc())
        raise e

    finally:
        if client is not None:
            client.emit('stop')


if __name__ == '__main__':
    main(*sys.argv[1:])
