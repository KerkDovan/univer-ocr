import random

from PIL import Image, ImageDraw, ImageFont


def generate(width, height):
    image = Image.new('RGB', (width, height), (200, 200, 200))
    mask = Image.new('L', (width, height))
    d_image = ImageDraw.Draw(image)

    text = 'Hello\nkeks!'
    font_size = 24
    font = ImageFont.truetype('arial.ttf', size=font_size)
    text_options = {
        'font': font,
    }
    t_width, t_height = font.getsize_multiline(text)

    x = random.randint(0, width - t_width * 2)
    y = random.randint(0, height - t_height * 2)
    d_image.text((x, y), text, fill=(0, 0, 0), **text_options)

    dx, dy = 0, 0
    for char in text:
        c_width, c_height = font.getsize(char)
        if char == '\n':
            dx = 0
            dy += font.getsize_multiline(f'A\nA')[1] - font.getsize('A')[1]
        else:
            d_image.text((x + dx + t_width, y + dy), char, fill=(0, 200, 0), **text_options)
            d_image.text((x + dx, y + dy + t_height), char, fill=(200, 0, 0), **text_options)
            dx += c_width

    return {
        'image': image,
        'mask': mask,
    }
