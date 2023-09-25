#!/usr/bin/env python3
#
# coding=utf-8
#
# Copyright (C) 2023 Kaalleen
#               Credits go to https://github.com/kaspar98/StringArt/
#               which is where much of this code is originated from
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
#
"""
Inkscape extension to generate stringart from an image.
"""
import base64
import os
import threading
from io import BytesIO
from math import atan2

import numpy as np
from inkex import Boolean, EffectExtension
from inkex import Image as InkexImage
from inkex import Layer, PathElement, TextElement, Transform, Tspan, errormsg
from PIL import Image

from lib.draw import ellipse_perimeter, line_aa


class StringArt(EffectExtension):
    """
    Main class to parse arguments and image
    """

    def __init__(self, *args, **kwargs):
        EffectExtension.__init__(self, *args, **kwargs)

        self.line_order = {}
        self.stringart_layer = None
        self.shape = None
        self.nails = None

    def add_arguments(self, pars):
        # tabs
        pars.add_argument("--tabs", type=str, default=None, dest="tabs")
        pars.add_argument("--settings-tab", type=str, default=None, dest="content_tab")
        pars.add_argument("--about-tab", type=str, default=None, dest="content_tab")
        # arguments
        pars.add_argument('-s', '--stroke_width', type=float, dest="stroke_width", default=0.1)
        pars.add_argument('-l', '--num-lines', type=int, dest="num_lines", default=0)
        pars.add_argument('-r', '--random_nails', type=int, dest="random_nails", default=0)
        pars.add_argument('-n', '--nail-dist', type=int, dest="nail_dist", default=4)
        pars.add_argument('--shape', type=str, dest="shape", default="circle")
        pars.add_argument('--wb', type=Boolean, dest="wb", default=False)
        pars.add_argument('--color-mode', type=str, dest="color_mode", default="black")
        pars.add_argument('--output-nail-numbers', type=Boolean, dest="output_nail_numbers",
                          default=False)
        pars.add_argument('--output-nail-order', type=Boolean, dest="output_nail_order",
                          default=False)

    def effect(self):
        images = [image for image in self.svg.selection if image.TAG == "image"]
        if not images:
            errormsg("Please select at least one image.")
            return

        self.stringart_layer = Layer(self.svg.get_unique_id('stringart_'))
        self.stringart_layer.label = "String Art"
        self.svg.append(self.stringart_layer)

        for image in images:
            width = int(float(image.get('width', '64')))
            height = int(float(image.get('height', '64')))

            image_byte_string = self._get_image_byte_string(image)
            if not image_byte_string:
                continue
            with Image.open(image_byte_string) as image:
                img = image.resize((width, height))
                img = img.transpose(Image.FLIP_TOP_BOTTOM)
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                img = img.transpose(Image.TRANSVERSE)
                if self.options.color_mode == 'hsv':
                    img = img.convert('HSV')
                else:
                    img = img.convert('RGB')

            self._process_image(img)

            if self.options.output_nail_numbers:
                self._insert_nail_numbers()

            if self.options.output_nail_order:
                self._insert_nail_order_text()

    def _get_image_byte_string(self, image):
        if isinstance(image, InkexImage):
            image = image.get('xlink:href', None)
            if not image:
                return None
            # linked image
            if image.startswith('file'):
                image_string = image[7:]
                if not os.path.isfile(image_string):
                    return None
            # embedded imagae
            elif image.startswith('data'):
                image = image.split(',', 1)
                image_string = BytesIO(base64.b64decode(image[1]))
        return image_string

    def _create_art(self, color, orig_pic, stroke_width):
        str_pic = self._init_canvas()
        current_position = self.nails[0]
        pull_order = [0]

        i = 0
        fails = 0
        while True:
            i += 1

            if self.options.num_lines == 0:
                if fails >= 3:
                    break
            else:
                if i > self.options.num_lines:
                    break

            idx, best_nail_position, best_cumulative_improvement = find_best_nail_position(
                pull_order[-1],
                current_position,
                self.nails,
                str_pic,
                orig_pic,
                stroke_width,
                self.options.random_nails)

            if best_cumulative_improvement <= 0:
                fails += 1
                continue

            pull_order.append(idx)
            best_overlayed_line, rows, columns = get_aa_line(
                current_position,
                best_nail_position,
                stroke_width,
                str_pic)
            str_pic[rows, columns] = best_overlayed_line

            current_position = best_nail_position

        self.line_order[color] = pull_order

    def _create_circle_nail_positions(self):
        height, width = self.shape

        r1_multip = 1
        r2_multip = 1
        if self.options.shape == 'ellipse':
            if height / width > 1:
                r1_multip = height / width
            elif width / height > 1:
                r2_multip = width / height

        nail_dist = self.options.nail_dist

        centre = (height // 2, width // 2)
        radius = min(height, width) // 2 - 1
        rows, columns = ellipse_perimeter(
            centre[0],
            centre[1],
            int(radius*r1_multip),
            int(radius*r2_multip))
        nails = list(set((rows[i], columns[i]) for i in range(len(columns))))
        nails.sort(key=lambda c: atan2(c[0] - centre[0], c[1] - centre[1]))
        nails = nails[::nail_dist]

        return np.asarray(nails)

    def _create_rectangle_nail_positions(self):
        height, width = self.shape
        nail_dist = self.options.nail_dist

        nails_top = [(0, i) for i in range(0, width, nail_dist)]
        nails_bot = [(height-1, i) for i in range(0, width, nail_dist)]
        nails_right = [(i, width-1) for i in range(1, height-1, nail_dist)]
        nails_left = [(i, 0) for i in range(1, height-1, nail_dist)]
        nails = nails_top + nails_right + nails_bot + nails_left

        return np.array(nails)

    def _init_canvas(self):
        if self.options.wb:
            return np.zeros(self.shape)
        return np.ones(self.shape)

    def _insert_element(self, pull_order, color="black"):
        style = f"fill:none;stroke:{color};stroke-width:{self.options.stroke_width}"
        path = "M "
        for nail in pull_order:
            point = self.nails[nail]
            path += f"{point[0]}, {point[1]} "
        element = PathElement.new(
            path=path,
            style=style)
        self.stringart_layer.append(element)

    def _process_image(self, image):
        stroke_width = self.options.stroke_width
        if not self.options.wb:
            stroke_width *= -1

        img = np.array(image)
        if np.any(img > 100):
            img = img / 255

        self.shape = (len(img), len(img[0]))

        if self.options.shape == 'rect':
            self.nails = self._create_rectangle_nail_positions()
        else:
            self.nails = self._create_circle_nail_positions()

        if self.options.color_mode == 'rgb':
            self._generate_rgb_line_order(img, stroke_width)
        else:
            self._generate_grayscale_line_order(img, stroke_width)

        for color, line_order in self.line_order.items():
            self._insert_element(line_order, color=color)

    def _generate_rgb_line_order(self, img, stroke_width):
        blue = img[:, :, 0]
        red = img[:, :, 1]
        green = img[:, :, 2]

        # creating threads
        thread1 = threading.Thread(target=self._create_art, name="thread1",
                                   args=['red', red, stroke_width])
        thread2 = threading.Thread(target=self._create_art, name="thread2",
                                   args=['green', green, stroke_width])
        thread3 = threading.Thread(target=self._create_art, name="thread3",
                                   args=['blue', blue, stroke_width])

        thread1.start()
        thread2.start()
        thread3.start()

        thread1.join()
        thread2.join()
        thread3.join()

    def _generate_grayscale_line_order(self, img, stroke_width):
        orig_pic = rgb2gray(img) * 0.9

        if self.options.wb:
            self._create_art('white', orig_pic, stroke_width / 2)
        else:
            self._create_art('black', orig_pic, stroke_width / 2)

    def _insert_nail_numbers(self):
        center_x, center_y = self.stringart_layer.bounding_box().center
        scale_x, scale_y = (1.05, 1.05)
        scale_matrix = f"matrix({ scale_x }, 0, 0, { scale_y }," \
                       f"{ center_x - scale_x * center_x}," \
                       f"{ center_y - scale_y * center_y})"
        text_element = TextElement(transform=str(Transform(scale_matrix)),
                                   style="font-size: 2px;text-anchor: middle;")
        for i, nail in enumerate(self.nails[:-4:5]):
            text = Tspan(str(i * 5 + 1), x=str(nail[0]), y=str(nail[1]))
            text_element.append(text)
        self.stringart_layer.append(text_element)

    def _insert_nail_order_text(self):
        page_num = 1
        page_bbox = self.svg.get_page_bbox()
        font_size = 10

        for _color, line_order in self.line_order.items():
            page = self._generate_new_page(page_bbox)
            page_num += 1
            page_bbox = page.bounding_box

            x_pos = page_bbox.left + 40
            y_pos = 20 + font_size
            style = f"font-size:{ font_size }px;line-height:1;text-align: end;text-anchor: end;"
            text_element = TextElement(x=str(x_pos), y=str(y_pos), style=style)
            text_element.set('xml:space', 'preserve')
            self.stringart_layer.append(text_element)
            for nail in line_order:
                text = Tspan(str(nail + 1))
                text.set('x', str(x_pos))
                text.set('y', str(y_pos))
                text.set('sodipodi:role', 'line')
                text_element.append(text)

                y_pos += 10

                # give ti some extra space until inkex handles the text bounding boxes better
                # or I handle units better
                if text.bounding_box().bottom + 15 > page.bounding_box.bottom:
                    x_pos += 50
                    y_pos = 20 + font_size

                    if x_pos + 15 > page.bounding_box.right:
                        page = self._generate_new_page(page_bbox)
                        page_bbox = page.bounding_box
                        x_pos += 20

                    text_element = TextElement(x=str(x_pos), y=str(y_pos), style=style)
                    self.stringart_layer.append(text_element)

    def _generate_new_page(self, bbox):
        namedview = self.svg.namedview
        page = namedview.new_page(
            str(bbox.right + 10),
            str(0),
            str(bbox.width),
            str(bbox.height)
        )
        return page


def rgb2gray(rgb):
    """
    rgb to grayscale
    """
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


def get_aa_line(from_pos, to_pos, stroke_width, picture):
    """
    Returns anti-aliased line pixel coordinates
    """
    rows, columns, val = line_aa(from_pos[0], from_pos[1], to_pos[0], to_pos[1])
    line = picture[rows, columns] + stroke_width * val
    line = np.clip(line, a_min=0, a_max=1)

    return line, rows, columns


def find_best_nail_position(prev_nail,  # pylint: disable=too-many-arguments, too-many-locals
                            current_position, nails, str_pic, orig_pic, stroke_width, random_nails):
    """
    Returns values to determine best choice for the next line
    """
    best_cumulative_improvement = -99999
    best_nail_position = None
    best_nail_idx = None

    if random_nails != 0:
        nail_ids = np.random.choice(range(len(nails)), size=random_nails, replace=False)
        nails_and_ids = list(zip(nail_ids, nails[nail_ids]))
    else:
        nails_and_ids = enumerate(nails)

    for nail_idx, nail_position in nails_and_ids:
        if nail_idx == prev_nail:
            continue

        overlayed_line, rows, columns = get_aa_line(
            current_position,
            nail_position,
            stroke_width,
            str_pic
        )

        before_overlayed_line_diff = np.abs(str_pic[rows, columns] - orig_pic[rows, columns])**2
        after_overlayed_line_diff = np.abs(overlayed_line - orig_pic[rows, columns])**2

        cumulative_improvement = np.sum(before_overlayed_line_diff - after_overlayed_line_diff)

        if cumulative_improvement >= best_cumulative_improvement:
            best_cumulative_improvement = cumulative_improvement
            best_nail_position = nail_position
            best_nail_idx = nail_idx

    return best_nail_idx, best_nail_position, best_cumulative_improvement


if __name__ == '__main__':
    StringArt().run()
