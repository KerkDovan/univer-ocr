from web_app import create_app, socketio

app = create_app()


if __name__ == '__main__':
    socketio.run(app, host='127.0.0.1', port=80, debug=True)
