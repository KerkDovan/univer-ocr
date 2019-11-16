from collections import namedtuple

from flask import Blueprint, render_template, send_file

from ..components import image_generator as ig

main_bp = Blueprint('main_bp', __name__)

generated = None

Font = namedtuple('Font', 'name normal bold italic bold_italic')
used_fonts = [
    Font('Arial', 'arial.ttf', 'arialbd.ttf', 'ariali.ttf', 'arialbi.ttf'),
    Font('Arial Narrow', 'ARIALN.TTF', 'ARIALNB.TTF', 'ARIALNI.TTF', 'ARIALNBI.TTF'),
    Font('Calibri', 'calibri.ttf', 'calibrib.ttf', 'calibrii.ttf', 'calibriz.ttf'),
    Font('Calibri Light', 'calibril.ttf', None, 'calibrili.ttf', None),
    Font('Cambria', 'cambria.ttc', 'cambriab.ttf', 'cambriai.ttf', 'cambriaz.ttf'),
    Font('Comic Sans', 'comic.ttf', 'comicbd.ttf', 'comici.ttf', 'comicz.ttf'),
    Font('Consolas', 'consola.ttf', 'consolab.ttf', 'consolai.ttf', 'consolaz.ttf'),
    Font('Times New Roman', 'times.ttf', 'timesbd.ttf', 'timesi.ttf', 'timesbi.ttf'),
    Font('Verdana', 'verdana.ttf', 'verdanab.ttf', 'verdanai.ttf', 'verdanaz.ttf'),
]


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


@main_bp.route('/fonts')
def fonts():
    context = {
        'fonts': used_fonts,
    }
    return render_template('fonts.html', **context)
