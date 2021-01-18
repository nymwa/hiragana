import random as rd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image, ImageOps
from svgwrite import Drawing
from cairosvg import svg2png

class CubicBezier:
    def __init__(self, mx, my, c1x, c1y, c2x, c2y, c3x, c3y):
        self.mx = mx
        self.my = my
        self.c1x = c1x
        self.c1y = c1y
        self.c2x = c2x
        self.c2y = c2y
        self.c3x = c3x
        self.c3y = c3y

    def __str__(self):
        return 'M{},{} C{},{} {},{} {},{}'.format(self.mx, self.my, self.c1x, self.c1y, self.c2x, self.c2y, self.c3x, self.c3y)

def sample_bezier():
    mx, my, c1x, c1y, c2x, c2y, c3x, c3y = [rd.randrange(1, 8) * 4 for _ in range(8)]
    return CubicBezier(mx, my, c1x, c1y, c2x, c2y, c3x, c3y)

def make_curve_list():
    num_curves = rd.randrange(1, 5)
    return [sample_bezier() for _ in range(num_curves)]

def generate_svg(curve_list, sample_width = False):
    svg = Drawing(size = (32, 32))
    for curve in curve_list:
        if sample_width:
            width = rd.randrange(4, 12) / 8
        else:
            width = 1
        svg.add(svg.path(d=str(curve), stroke='black', fill='none', stroke_width=width))
    return svg

def to_img(svg):
    png = svg2png(bytestring = svg.tostring())
    img = Image.open(BytesIO(png))
    _, _, _, img = img.split()
    return np.array(img)

def noising(arr):
    width = rd.randrange(32)
    arr = arr + (np.random.randn(32, 32) - 0.5) * width
    arr = np.clip(arr, 0, 255)
    return arr

def normalize(arr):
    return (arr / 255).astype('float32')

def binarization(arr):
    return (arr > 128).astype('float32')

def make_src(curve_list):
    return normalize(noising(to_img(generate_svg(curve_list, sample_width=True))))

def make_trg(curve_list):
    return binarization(to_img(generate_svg(curve_list)))
        
