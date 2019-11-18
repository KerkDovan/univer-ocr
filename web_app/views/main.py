from datetime import datetime

from flask import redirect, render_template, request, send_file, url_for

from ..components import image_generator as ig
from ..components.primitives import CHARS, FONTS_LIST, encode_char
from . import main_bp

raw, demo = None, None


def generate():
    global raw, demo
    raw, demo = ig.generate(1920, 1080, True)


@main_bp.route('/')
@main_bp.route('/index')
def index():
    return render_template('index.html')


@main_bp.route('/generate_new')
def generate_new():
    generate()
    return redirect(request.referrer or url_for('main_bp.index'))


@main_bp.route('/view_layers/<mode>')
def view_layers(mode):
    ts = datetime.now()
    if demo is None or raw is None:
        generate()
    images = demo if mode == 'demo' else raw
    context = {
        'time_consumed': datetime.now() - ts,
        'mode': mode,
        'layer_names': list(images.keys()),
    }
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
    return render_template('test-nn.html')
