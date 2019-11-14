import random
import string
from collections import namedtuple

from PIL import Image, ImageDraw, ImageFont


def generate(width, height):
    image_tuple = namedtuple(
        'image_tuple', 'image line_top line_center line_bottom separator char_box')

    image = image_tuple(
        image=Image.new('RGBA', (width, height), (200, 200, 200, 255)),
        line_top=Image.new('RGBA', (width, height)),
        line_center=Image.new('RGBA', (width, height)),
        line_bottom=Image.new('RGBA', (width, height)),
        separator=Image.new('RGBA', (width, height)),
        char_box=Image.new('RGBA', (width, height)),
    )
    draw = image_tuple(**{
        x: ImageDraw.ImageDraw(image[i])
        for i, x in enumerate(image._fields)}
    )

    fontname = 'verdana.ttf'
    basefontsize = 40
    punct1 = u' '.join(string.punctuation[:len(string.punctuation) // 2])
    punct2 = u' '.join(string.punctuation[len(string.punctuation) // 2:])
    text = [
        (u'А Б В Г Д Е Ё Ж З И Й К Л М Н О П', ImageFont.truetype(fontname, basefontsize)),
        (u'Р С Т У Ф Х Ц Ч Ш Щ Ъ Ы Ь Э Ю Я', ImageFont.truetype(fontname, basefontsize)),
        (u'а б в г д е ё ж з и й к л м н о п', ImageFont.truetype(fontname, basefontsize + 4)),
        (u'р с т у ф х ц ч ш щ ъ ы ь э ю я', ImageFont.truetype(fontname, basefontsize + 4)),
        (u'A B C D E F G H I J K L M', ImageFont.truetype(fontname, basefontsize + 8)),
        (u'N O P Q R S T U V W X Y Z', ImageFont.truetype(fontname, basefontsize + 8)),
        (u'a b c d e f g h i j k l m', ImageFont.truetype(fontname, basefontsize + 14)),
        (u'n o p q r s t u v w x y z', ImageFont.truetype(fontname, basefontsize + 14)),
        (punct1, ImageFont.truetype(fontname, basefontsize)),
        (punct2, ImageFont.truetype(fontname, basefontsize)),
        (u'0 1 2 3 4 5 6 7 8 9', ImageFont.truetype(fontname, basefontsize)),
    ]
    t_width, t_height = 0, 0
    for line, font in text:
        tw, th = font.getsize(line)
        t_width = max(t_width, tw)
        t_height += th

    x = random.randint(0, width - t_width)
    y = random.randint(0, height - t_height)

    dy = 0
    for line, font in text:
        lw, lh = font.getsize(line)
        lh2, lh4 = lh // 2, lh // 4
        lh34 = lh2 + lh4

        dx = 0
        for i, char in enumerate(line):
            c_width, c_height = font.getsize(line[i])
            w, h = font.getmask(char).size
            if w != c_width and i == 0:
                dx = w - c_width

            right, bottom = font.getsize(line[:i + 1])
            right += x
            bottom = min(c_height, bottom) + y + dy
            top = bottom - h
            left = right - c_width

            if not char.isspace():
                draw.image.text(
                    (left - dx + (w - c_width), y + dy), char, fill=(0, 0, 0), font=font)
                draw.char_box.rectangle(
                    (left - dx, top, right, bottom), fill=(0, 0, 200, 50))

            if i == len(line) - 1:
                continue

            draw.separator.rectangle(
                (right - 2, y + dy, right + 2, y + dy + lh), fill=(200, 0, 200, 50))

        draw.line_top.rectangle(
            (x, y + dy, x + lw, y + dy + lh2), fill=(200, 0, 0, 50))
        draw.line_center.rectangle(
            (x, y + dy + lh4, x + lw, y + dy + lh34), fill=(200, 200, 0, 50))
        draw.line_bottom.rectangle(
            (x, y + dy + lh2, x + lw, y + dy + lh), fill=(0, 200, 0, 50))

        dy += font.getsize_multiline(f'\nA')[1] - font.getsize('A')[1]

    return {
        x: image[i] for i, x in enumerate(image._fields)
    }
