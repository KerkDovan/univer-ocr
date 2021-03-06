from datetime import datetime

from flask import redirect, render_template, request, send_file, url_for

from ..components import image_generator as ig
from ..components.interpreter import interpret
from ..components.primitives import CHARS, FONTS_LIST, encode_char
from . import main_bp
from .test_nn_ws import test_scripts

raw, demo = None, None
generation_time = datetime.now() - datetime.now()


def generate_demo():
    global raw, demo, generation_time
    ts = datetime.now()
    raw, demo = ig.generate_demo(1920, 1080)
    generation_time = datetime.now() - ts


def reset_generation_time():
    global generation_time
    generation_time = datetime.now() - datetime.now()


@main_bp.route('/')
@main_bp.route('/index')
def index():
    return render_template('index.html')


@main_bp.route('/generate_new')
def generate_new():
    generate_demo()
    return redirect(request.referrer or url_for('main_bp.index'))


@main_bp.route('/view_layers/<mode>')
def view_layers(mode):
    if demo is None or raw is None:
        generate_demo()
    images = demo if mode == 'demo' else raw
    context = {
        'time_consumed': generation_time,
        'mode': mode,
        'layer_names': list(images.keys()),
    }
    reset_generation_time()
    return render_template('view_layers.html', **context)


@main_bp.route('/image/<mode>/<image_type>')
def image(mode, image_type):
    im = (demo if mode == 'demo' else raw)[image_type]
    return send_file(ig.to_bytesio(im), mimetype='image/png', cache_timeout=0)


@main_bp.route('/chars')
def chars():
    context = {
        'chars': {x: encode_char(x) for x in CHARS},
    }
    return render_template('chars.html', **context)


@main_bp.route('/fonts')
def fonts():
    context = {
        'fonts': FONTS_LIST,
    }
    return render_template('fonts.html', **context)


@main_bp.route('/test-nn')
def test_nn():
    context = {
        'tests': test_scripts.items(),
    }
    return render_template('test-nn.html', **context)


@main_bp.route('/train')
def train():
    return render_template('train.html')


@main_bp.route('/interpret_data')
def interpret_data():
    if raw is None:
        generate_demo()
    ts = datetime.now()
    data = interpret(raw)
    context = {
        'time_consumed': generation_time + datetime.now() - ts,
        'data': data,
    }
    reset_generation_time()
    return render_template('interpret_data.html', **context)
