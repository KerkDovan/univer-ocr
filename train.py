import sys
import traceback

from socketIO_client import BaseNamespace, SocketIO

from web_app.components.my_model.train import init_emitter, train_model


def main(use_gpu=False):
    client = None
    try:
        socketIO = SocketIO('127.0.0.1', 80, Namespace=BaseNamespace, wait_for_connection=False)
        client = socketIO.define(BaseNamespace, '/train-ws')
        init_emitter(client)

    except Exception:
        print(f'Cannot connect to socket.io server, running in console mode')

    try:
        train_model(use_gpu == 'True' or use_gpu is True)

    except KeyboardInterrupt:
        print(f'Stopped by keyboard interrupt')

    except Exception:
        print(traceback.format_exc())

    finally:
        if client is not None:
            client.emit('stop')


if __name__ == '__main__':
    main(*sys.argv[1:])
