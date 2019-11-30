import sys

from socketIO_client import SocketIO, BaseNamespace

from web_app.components.my_model.train import train_model, init_emitter


def main(use_gpu=False):
    socketIO = SocketIO('127.0.0.1', 80, Namespace=BaseNamespace)
    client = socketIO.define(BaseNamespace, '/train-ws')
    init_emitter(client)

    train_model(use_gpu == 'True')


if __name__ == '__main__':
    main(*sys.argv[1:])
