from flask import render_template, send_file

from ..components import image_generator as ig
from ..components.primitives import CHARS, FONTS_LIST, encode_char
from . import main_bp

generated = None


@main_bp.route('/')
@main_bp.route('/index')
def index():
    global generated
    generated = ig.generate(1920, 1080, True)[1]
    context = {
        'layer_names': list(generated.keys()),
    }
    return render_template('index.html', **context)


@main_bp.route('/image/<image_type>')
def image(image_type):
    return send_file(
        ig.to_bytesio(generated[image_type]),
        mimetype='image/png',
        cache_timeout=0)


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
    return render_template('test-nn.html')
