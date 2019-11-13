from flask import Blueprint, render_template, send_file

from web_app import image_generator as ig

main_bp = Blueprint('main_bp', __name__)
generated = None


@main_bp.route('/')
@main_bp.route('/index')
def index():
    global generated
    generated = ig.generate(800, 800)
    return render_template('index.html')


@main_bp.route('/image/<image_type>')
def image(image_type):
    return send_file(
        ig.to_bytesio(generated[image_type]),
        mimetype='image/png',
        cache_timeout=0)
