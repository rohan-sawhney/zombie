#!/usr/local/bin/python3

from argparse import ArgumentParser, ArgumentTypeError
import glob
from imageio import imread, imwrite
import numpy as np
import os
import sys

def image_to_pfm(image_path, pfm_path, normalize = False, flip_y = False, scale = 1.0, offset = 0.0):
    '''
    Converts an image into a PFM file.

    Parameters
    -----------

    image_path: string 

    pfm_path: string

    normalize: bool
        normalize pixel values by dividing by 255
    
    flip_y: bool
        flip vertical orientation
    
    scale: float
    
    offset: floatt
    '''
    data = np.array(imread(image_path), dtype=np.float32)
    data = np.flip(data, axis=0)
    data = data / 255.0 if normalize else data
    data = data[:, :, :3] * scale - offset
    imwrite(pfm_path, data)

if __name__ == '__main__':
    parser = ArgumentParser(description='Convert image to PFM format.')
    parser.add_argument('images', type=str, nargs='+', help='List of input files to convert')
    parser.add_argument('--normalize', action='store_true', help='Map values to 0 to 1 range by dividing 255')
    parser.add_argument('--flip_y', action='store_true', help='Flip PFM along vertical direction before saving')
    parser.add_argument('--scale', type=float, help='Scale pixel values', default=1.0)
    parser.add_argument('--offset', type=float, help='Offset pixel values', default=0.0)

    args = parser.parse_args()

    images = []
    for pattern in args.images:
        images.extend(glob.glob(pattern))

    for image in images:
        output = os.path.splitext(image)[0] + '.pfm'
        os.makedirs(os.path.dirname(output), exist_ok=True)
        image_to_pfm(image, output, args.normalize, args.flip_y, args.scale, args.offset)

