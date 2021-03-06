import random
from textwrap import wrap

import numpy as np
from PIL import Image, ImageDraw

from faker import Faker

from ..primitives import BITS_COUNT, CHARS, FONTS_LIST, encode_char


class LayeredImage:
    layer_names = [
        'image',
        'image_monochrome',
        'paragraph',
        'line_top',
        'line_center',
        'line_bottom',
        'letter_spacing',
        'char_mask_box',
        'char_full_box',
        *[f'bit_{i}' for i in range(BITS_COUNT)],
    ]
    colors = {
        'image': (0, 0, 0, 255),
        **{layer: 255 for layer in layer_names[1:]},
    }
    colors_demo = {
        'image': (0, 0, 0, 255),
        'paragraph': (0, 0, 200, 50),
        'line_top': (200, 0, 0, 100),
        'line_center': (0, 0, 200, 150),
        'line_bottom': (0, 200, 0, 100),
        'letter_spacing': (200, 0, 200, 100),
        'char_mask_box': (200, 200, 0, 100),
        'char_full_box': (200, 200, 0, 100),
        **{f'bit_{i}': (200, 200, 0, 100) for i in range(BITS_COUNT)},
    }

    def __init__(self, width, height, bg_color, use_demo=False):
        self.width, self.height = width, height
        self.bg_color = bg_color
        self.use_demo = use_demo
        self.layers = {
            'image': Image.new('RGBA', (self.width, self.height), self.bg_color),
            **{
                name: Image.new('L', (self.width, self.height))
                for name in self.layer_names[1:]
            }
        }
        self.mask = None
        self._updage_mask()
        self.draw = {
            name: ImageDraw.ImageDraw(layer)
            for name, layer in self.layers.items()
        }
        self.demo = {
            'image': Image.new('RGBA', (self.width, self.height), self.bg_color),
            'guidelines': Image.new('RGBA', (self.width, self.height)),
            **{
                name: Image.new('RGBA', (self.width, self.height))
                for name in self.layer_names[1:]
            }
        } if self.use_demo else {}
        self.draw_demo = {
            name: ImageDraw.ImageDraw(layer)
            for name, layer in self.demo.items()
        }
        self.paragraphs_added = 0

    def get_raw(self):
        return self.layers

    def get_demo(self):
        return self.demo

    def rotate(self, angle):
        for images_set in [self.layers, self.demo]:
            for name, image in images_set.items():
                bg_color = self.bg_color if image.mode == 'RGBA' else 0
                rot = image.convert('RGBA').rotate(
                    angle, resample=Image.BILINEAR, expand=True)
                fff = Image.new('RGBA', rot.size, bg_color)
                res = Image.composite(rot, fff, rot).convert(image.mode)
                images_set[name] = res
        self.width, self.height = self.layers['image'].size
        return self

    def make_divisible_by(self, x, y):
        to_add_x = x - self.width % x
        to_add_y = y - self.height % y
        new_size = (self.width + to_add_x, self.height + to_add_y)
        pos = (to_add_x // 2, to_add_y // 2)
        for images_set in [self.layers, self.demo]:
            for name, image in images_set.items():
                bg_color = self.bg_color if image.mode == 'RGBA' else 0
                new_image = Image.new(image.mode, new_size, bg_color)
                new_image.paste(image, pos)
                images_set[name] = new_image
        return self

    def add_paragraph(self, text, font):
        spacing = font.size // 2
        ascent, descent = font.getmetrics()
        M_height, x_height = font.getmask('M').size[1], font.getmask('x').size[1]
        height = font.font.getsize(CHARS)[0][1]

        t_width, t_height = 0, 0
        for line in text:
            offset_x, _ = font.getoffset(line + CHARS)
            t_width = max(t_width, font.font.getsize(line)[0][0] + offset_x)
            t_height += font.getsize_multiline(
                f'{line}\nA', spacing=spacing
            )[1] - font.getsize('A')[1]

        margin = 3
        margin2 = 2 * margin
        ones = np.ones((t_height + margin2, t_width + margin2), dtype=np.uint8)
        x, y = None, None
        retries = 0
        while True:
            left_margin = 20
            rand_width = self.width - (t_width + margin2) - left_margin
            rand_height = self.height - (t_height + margin2)
            if rand_width < left_margin or rand_height < 0:
                # print(f'Paragraph is too big for the image')
                return
            x = random.randint(left_margin, rand_width)
            y = random.randint(0, rand_height)
            if np.sum(ones * self.mask[y:y + t_height + margin2, x:x + t_width + margin2]) == 0:
                break
            if retries > 100:
                # print(f'Number of retries exceeded')
                return
            retries += 1
        self.paragraphs_added += 1
        x, y = x + margin, y + margin

        self._paragraph((x, y, x + t_width, y + t_height))
        self._updage_mask()

        dy = 0
        for line in text:
            width = font.font.getsize(line)[0][0]
            offset_x, offset_y = font.getoffset(line + CHARS)

            left = x + offset_x
            right = left + width + offset_x

            y_ascent = y + dy + offset_y
            y_baseline = y_ascent + height - descent
            y_M = y_baseline - M_height
            y_x = y_baseline - x_height
            y_descent = y_baseline + descent

            _, lh = font.getsize(line)
            self._line(left, right, y_ascent, y_M, y_x, y_baseline, y_descent)

            for i, char in enumerate(line):
                c_width, c_height = font.getsize(line[i])
                w, h = font.getmask(char).size

                ch_r, ch_b = font.getsize(line[:i + 1])
                ch_r += offset_x
                ch_b = min(c_height, ch_b) + y + dy
                ch_t = ch_b - h
                ch_l = ch_r - c_width
                ch_offset_x = font.getoffset(char)[0]
                w10 = max(1, c_width / 10)

                self._char(char, (x + ch_l - ch_offset_x, y + dy), font)
                self._mask_box(char, (x + ch_l, ch_t, x + ch_r, ch_b))
                self._full_box(char, (
                    x + ch_l - ch_offset_x + w10,
                    y_ascent,
                    x + ch_r - ch_offset_x - w10,
                    y_descent))

                if i == len(line) - 1:
                    continue

                self._letter_spacing((
                    x + ch_r - ch_offset_x - w10,
                    y_ascent,
                    x + ch_r - ch_offset_x + w10,
                    y_descent))

            dy += font.getsize_multiline(f'{line}\nA', spacing=spacing)[1] - font.getsize('A')[1]

    def _paragraph(self, coords):
        self.draw['paragraph'].rectangle(coords, fill=self.colors['paragraph'])
        if self.use_demo:
            self.draw_demo['paragraph'].rectangle(
                coords, fill=self.colors_demo['paragraph'])

    def _char(self, char, position, font):
        self.draw['image'].text(position, char, fill=self.colors['image'], font=font)
        self.draw['image_monochrome'].text(
            position, char, fill=self.colors['image_monochrome'], font=font)
        if self.use_demo:
            self.draw_demo['image'].text(
                position, char, fill=self.colors_demo['image'], font=font)

    def _mask_box(self, char, coords):
        self.draw['char_mask_box'].rectangle(coords, fill=self.colors['char_mask_box'])
        if self.use_demo:
            self.draw_demo['char_mask_box'].rectangle(
                coords, fill=self.colors_demo['char_mask_box'])

    def _full_box(self, char, coords):
        bits = encode_char(char)
        self.draw['char_full_box'].rectangle(coords, fill=self.colors['char_full_box'])
        for i, bit in enumerate(bits):
            if bit == '0':
                continue
            self.draw[f'bit_{i}'].rectangle(coords, fill=self.colors[f'bit_{i}'])
        if self.use_demo:
            self.draw_demo['char_full_box'].rectangle(
                coords, fill=self.colors_demo['char_full_box'])
            for i, bit in enumerate(bits):
                if bit == '0':
                    continue
                self.draw_demo[f'bit_{i}'].rectangle(coords, fill=self.colors_demo[f'bit_{i}'])

    def _letter_spacing(self, coords):
        self.draw['letter_spacing'].rectangle(coords, fill=self.colors['letter_spacing'])
        if self.use_demo:
            self.draw_demo['letter_spacing'].rectangle(
                coords, fill=self.colors_demo['letter_spacing'])

    def _line(self, left, right, y_ascent, y_M, y_x, y_baseline, y_descent):
        line_top_coords = (left, y_ascent, right, y_baseline)
        line_center_coords = (left, y_x, right, y_baseline)
        line_bottom_coords = (left, y_x, right, y_descent)

        self.draw['line_top'].rectangle(line_top_coords, fill=self.colors['line_top'])
        self.draw['line_center'].rectangle(line_center_coords, fill=self.colors['line_center'])
        self.draw['line_bottom'].rectangle(line_bottom_coords, fill=self.colors['line_bottom'])

        if self.use_demo:
            def hline(y, color):
                self.draw_demo['guidelines'].line((left, y, right, y), fill=color, width=1)

            hline(y_ascent, (200, 0, 200))
            hline(y_M, (0, 200, 0))
            hline(y_x, (0, 200, 200))
            hline(y_baseline, (200, 0, 0))
            hline(y_descent, (0, 0, 200))

            self.draw_demo['line_top'].rectangle(
                line_top_coords, fill=self.colors_demo['line_top'])
            self.draw_demo['line_center'].rectangle(
                line_center_coords, fill=self.colors_demo['line_center'])
            self.draw_demo['line_bottom'].rectangle(
                line_bottom_coords, fill=self.colors_demo['line_bottom'])

    def _updage_mask(self):
        self.mask = np.array(self.layers['paragraph'])


def random_font(min_size=12, max_size=48):
    style = random.choice(['normal', 'bold'])
    font = None
    while font is None:
        font = getattr(random.choice(FONTS_LIST), style)
        font = font(size=random.randint(min_size, max_size))
    return font


def random_text(min_wrap=30, max_wrap=100):
    if True or np.random.choice([True, False], p=[0.1, 0.9]):
        text = ' '.join(
            ''.join(random.choice(CHARS[1:]) for i in range(random.randint(1, 10)))
            for j in range(random.randint(3, 30)))
    else:
        fake = Faker(random.choice(['la', 'en_US', 'ru_RU']))
        text = fake.paragraph(nb_sentences=random.randint(3, 10))
    return wrap(text, random.randint(min_wrap, max_wrap))


def generate_demo(width, height):
    layers = LayeredImage(width, height, (200, 200, 200, 255), use_demo=True)
    for _ in range(30):
        layers.add_paragraph(random_text(), random_font())
    return layers.get_raw(), layers.get_demo()
