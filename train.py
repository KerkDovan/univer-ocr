import sys
import traceback

from socketIO_client import BaseNamespace, SocketIO

from web_app.components.my_model.train import init_emitter, train_model


def main(use_gpu=False):
    socketIO = SocketIO('127.0.0.1', 80, Namespace=BaseNamespace)
    client = socketIO.define(BaseNamespace, '/train-ws')
    init_emitter(client)

    try:
        train_model(use_gpu == 'True' or use_gpu is True)

    except Exception:
        print(traceback.format_exc())

    finally:
        client.emit('stop')


if __name__ == '__main__':
    main(*sys.argv[1:])
