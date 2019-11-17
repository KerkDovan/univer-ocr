from flask import Flask
from flask_socketio import SocketIO

from .config import Config

socketio = SocketIO()


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    with app.app_context():
        from .views import main

        app.register_blueprint(main.main_bp)

        socketio.init_app(app)
        return app
