import importlib
import sys
from pathlib import Path

from flask import Flask

from .config import Config

sys.path.append(Path('..', 'image_generator'))
image_generator = importlib.import_module('image_generator')


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    with app.app_context():
        from .views import main

        app.register_blueprint(main.main_bp)

        return app
