"""Convert a renderer HDR capture to PNG using its exact display transfer (requires Pillow)."""
import argparse
from array import array
from pathlib import Path
import math
import struct
import sys
from PIL import Image

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('input', type=Path)
parser.add_argument('output', type=Path)
parser.add_argument('--exposure', type=float, default=1.0)
args = parser.parse_args()
raw = args.input.read_bytes()
width, height = struct.unpack_from('<II', raw)
values = array('f')
values.frombytes(raw[8:])
if sys.byteorder != 'little':
    values.byteswap()
if len(values) != width * height * 4 or not all(map(math.isfinite, values)):
    raise ValueError('Invalid HDR capture')
if not math.isfinite(args.exposure) or args.exposure <= 0:
    raise ValueError('Exposure must be finite and positive')
pixels = bytearray()
for index in range(0, len(values), 4):
    rgb = [max(0.0, x) * args.exposure for x in values[index:index+3]]
    scale = 1.0 / (1.0 + max(rgb))
    for x in rgb:
        x *= scale
        x = 12.92*x if x <= 0.0031308 else 1.055*x**(1.0/2.4)-0.055
        pixels.append(round(min(1.0, max(0.0, x))*255))
Image.frombytes('RGB', (width, height), bytes(pixels)).save(args.output)
