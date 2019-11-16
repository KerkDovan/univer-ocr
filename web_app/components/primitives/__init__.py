import string

from PIL.ImageFont import truetype

RUSSIAN_LOWERCASE = u'абвгдеёжзийклмнопрстуфхцчшщъыьэюя'
RUSSIAN_UPPERCASE = u'АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ'
RUSSIAN = RUSSIAN_LOWERCASE + RUSSIAN_UPPERCASE
CHARS = RUSSIAN + string.digits + string.ascii_letters + string.punctuation + ' '

PUNCTUATION_ENCODING_MAP = {
    '!': 'exclamation', '"': 'double_quote', '#': 'hash_sign', '$': 'dollar_sign',
    '%': 'percent', '&': 'ampersand', '\'': 'single_quote', '(': 'op_parentheses',
    ')': 'cl_parentheses', '*': 'asterisk', '+': 'plus', ',': 'comma', '-': 'minus',
    '.': 'point', '/': 'slash', ':': 'colon', ';': 'semicolon', '<': 'less', '=': 'equal',
    '>': 'more', '?': 'question', '@': 'at_sign', '[': 'op_square', '\\': 'backslash',
    ']': 'cl_square', '^': 'caret', '_': 'underscore', '`': 'apostrophe', '{': 'op_curly',
    '|': 'vertical_bar', '}': 'cl_curly', '~': 'tilde', ' ': 'space',
}
PUNCTUATION_DECODING_MAP = {value: key for key, value in PUNCTUATION_ENCODING_MAP.items()}


def encode_char(char):
    assert len(char) == 1
    if char not in CHARS:
        return 'unknown'
    if char not in string.punctuation + ' ':
        return char
    return PUNCTUATION_ENCODING_MAP.get(char, 'unknown')


def decode_char(encoded_char):
    if encode_char in CHARS:
        return encode_char
    return PUNCTUATION_DECODING_MAP.get(encode_char, 'unknown')


class Font:
    def __init__(self, name, normal, bold, italic, bold_italic):
        self.name = name
        self.normal_path = normal
        self.bold_path = bold
        self.italic_path = italic
        self.bold_italic_path = bold_italic

    def normal(self, size=10, index=0, encoding="", layout_engine=None):
        if self.normal_path is None:
            return None
        return truetype(font=self.normal_path, size=size, index=index,
                        encoding=encoding, layout_engine=layout_engine)

    def bold(self, size=10, index=0, encoding="", layout_engine=None):
        if self.bold_path is None:
            return None
        return truetype(font=self.bold_path, size=size, index=index,
                        encoding=encoding, layout_engine=layout_engine)

    def italic(self, size=10, index=0, encoding="", layout_engine=None):
        if self.italic_path is None:
            return None
        return truetype(font=self.italic_path, size=size, index=index,
                        encoding=encoding, layout_engine=layout_engine)

    def bold_italic(self, size=10, index=0, encoding="", layout_engine=None):
        if self.bold_italic_path is None:
            return None
        return truetype(font=self.bold_italic_path, size=size, index=index,
                        encoding=encoding, layout_engine=layout_engine)


FONTS_LIST = [
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
FONTS_DICT = {font.name: font for font in FONTS_LIST}
