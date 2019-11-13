import random
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

    text = 'こんにちは,\nHoàng Tùng Lâm!\nМеня зовут Kerk Dovan.'
    font_size = 42
    font = ImageFont.truetype('yumin.ttf', size=font_size)
    t_width, t_height = font.getsize_multiline(text)

    x = random.randint(0, width - t_width)
    y = random.randint(0, height - t_height)
    draw.image.text((x, y), text, fill=(0, 0, 0), font=font)

    dy = 0
    for line in text.split('\n'):
        lw, lh = font.getsize(line)
        lh2, lh4 = lh // 2, lh // 4
        lh34 = lh2 + lh4

        for i, char in enumerate(line):
            c_width, c_height = font.getsize(line[i])
            w, h = font.getmask(char).size

            right, bottom = font.getsize(line[:i + 1])
            right += x
            bottom = min(c_height, bottom) + y + dy
            top = bottom - h
            left = right - w

            if not char.isspace():
                draw.char_box.rectangle(
                    (left, top, right, bottom), fill=(0, 0, 200, 50))

            if i == len(line) - 1:
                continue

            draw.separator.rectangle(
                (right - w // 4, y + dy, right, y + dy + lh), fill=(200, 0, 200, 50))

        draw.line_top.rectangle(
            (x, y + dy, x + lw, y + dy + lh2), fill=(200, 0, 0, 50))
        draw.line_center.rectangle(
            (x, y + dy + lh4, x + lw, y + dy + lh34), fill=(200, 200, 0, 50))
        draw.line_bottom.rectangle(
            (x, y + dy + lh2, x + lw, y + dy + lh), fill=(0, 200, 0, 50))

        dy += font.getsize_multiline(f'\nA')[1] - font.getsize('A')[1]

    return {
        **{x: image[i] for i, x in enumerate(image._fields)}
    }
