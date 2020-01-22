import string
from math import ceil, log

from PIL.ImageFont import truetype

_______ids_______ = u'012345678901234567890123456789012'
RUSSIAN_LOWERCASE = u'абвгдеёжзийклмнопрстуфхцчшщъыьэюя'
RUSSIAN_UPPERCASE = u'АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ'
ENGLISH_LOWERCASE = u'abcdefghijklmnopqrstuvwxyz'
ENGLISH_UPPERCASE = u'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
RUSSIAN = RUSSIAN_LOWERCASE + RUSSIAN_UPPERCASE
ENGLISH = ENGLISH_LOWERCASE + ENGLISH_UPPERCASE
CHARS = '\t' + ' ' + RUSSIAN + string.digits + ENGLISH + string.punctuation
CHARS_IDS = {char: i for i, char in enumerate(CHARS)}

SIMILAR_CHARS_PAIRS_LIST = [
    # Lowercase
    (RUSSIAN_LOWERCASE[0], ENGLISH_LOWERCASE[0]),
    (RUSSIAN_LOWERCASE[5], ENGLISH_LOWERCASE[4]),
    (RUSSIAN_LOWERCASE[15], ENGLISH_LOWERCASE[14]),
    (RUSSIAN_LOWERCASE[17], ENGLISH_LOWERCASE[15]),
    (RUSSIAN_LOWERCASE[18], ENGLISH_LOWERCASE[2]),
    (RUSSIAN_LOWERCASE[20], ENGLISH_LOWERCASE[24]),
    (RUSSIAN_LOWERCASE[22], ENGLISH_LOWERCASE[23]),
    # Uppercase
    (RUSSIAN_UPPERCASE[0], ENGLISH_UPPERCASE[0]),
    (RUSSIAN_UPPERCASE[2], ENGLISH_UPPERCASE[1]),
    (RUSSIAN_UPPERCASE[5], ENGLISH_UPPERCASE[4]),
    (RUSSIAN_UPPERCASE[11], ENGLISH_UPPERCASE[10]),
    (RUSSIAN_UPPERCASE[13], ENGLISH_UPPERCASE[12]),
    (RUSSIAN_UPPERCASE[15], ENGLISH_UPPERCASE[14]),
    (RUSSIAN_UPPERCASE[14], ENGLISH_UPPERCASE[7]),
    (RUSSIAN_UPPERCASE[17], ENGLISH_UPPERCASE[15]),
    (RUSSIAN_UPPERCASE[18], ENGLISH_UPPERCASE[2]),
    (RUSSIAN_UPPERCASE[19], ENGLISH_UPPERCASE[19]),
    (RUSSIAN_UPPERCASE[22], ENGLISH_UPPERCASE[23]),
]
SIMILAR_CHARS = {
    k: v
    for v in SIMILAR_CHARS_PAIRS_LIST
    for k in v
}

BITS_COUNT = ceil(log(len(CHARS) + 1, 2))

ENCODING_MAP = {
    char: (bin(char_id)[2:][::-1] + '0' * BITS_COUNT)[:BITS_COUNT]
    for char_id, char in enumerate(CHARS)
}
DECODING_MAP = {encoded: char for char, encoded in ENCODING_MAP.items()}


def are_similar(char1, char2):
    return char1 in SIMILAR_CHARS.get(char2, ())


def encode_char(char):
    assert len(char) == 1
    return ENCODING_MAP.get(char, '1' * BITS_COUNT)


def decode_char(encoded):
    assert len(encoded) == BITS_COUNT and set(encoded) in [{'0'}, {'1'}, {'0', '1'}]
    return DECODING_MAP.get(encoded, 'unknown')


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
    Font('Comic Sans MS', 'comic.ttf', 'comicbd.ttf', 'comici.ttf', 'comicz.ttf'),
    Font('Consolas', 'consola.ttf', 'consolab.ttf', 'consolai.ttf', 'consolaz.ttf'),
    Font('Times New Roman', 'times.ttf', 'timesbd.ttf', 'timesi.ttf', 'timesbi.ttf'),
    Font('Verdana', 'verdana.ttf', 'verdanab.ttf', 'verdanai.ttf', 'verdanaz.ttf'),
]
FONTS_DICT = {font.name: font for font in FONTS_LIST}
