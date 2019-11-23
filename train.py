from socketIO_client import SocketIO, BaseNamespace

from web_app.components.my_model.train import train_model, init_emitter


def main():
    socketIO = SocketIO('127.0.0.1', 80, Namespace=BaseNamespace)
    client = socketIO.define(BaseNamespace, '/train-ws')
    init_emitter(client)

    train_model()


if __name__ == '__main__':
    main()
