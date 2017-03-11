#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import logging
import os

import cv2
from functions import *

"""
MC - D3 - QR detection & decoding
"""
_DEFAULT_OUTPUT_PATH = './Output/'
_DEFAULT_INPUT_PATH = './Resources/'


def main(args):
    # Load images
    images = load_images(args.input)

    for image_name, img in images.items():
        
        for rotated in range(2):
    
            if rotated == 0:
                _, mask = color_filter(img)   
                inclination_corrected = rotate(img, mask)
            else:
                inclination_corrected = img 
            
            # Obtener máscara de filtro de color
            color_filtered, mask = color_filter(inclination_corrected)
            
            # Cambiar a espacio de color escala de grises
            gray = cv2.cvtColor(inclination_corrected, cv2.COLOR_BGR2GRAY)
            
            # Detección de bordes
            edges = edge_detection(gray)
            for i in range(10,50,5):
                blocks = None
                try:
                    connected = connected_components(edges, mask, i)
                    qrCode, highlight = qrcode_detection(connected, inclination_corrected)
                    processed = qrcode_postprocess(qrCode)
                    contours, blocks = qrcode_orientationdetection(processed, image_name)
                except Exception:
                    continue
                    
                if blocks is not None:
                    reorientated_img = qrcode_reorientate(processed, contours, blocks)
                
                    cv2.imshow(image_name, resize(reorientated_img))
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                    break;
            
            if blocks is not None:
                break;
            
        if blocks is not None:
            print("=> {}\t\t{}".format(image_name, "FOUND"))
        else:
            print("=> {}\t\t{}".format(image_name, "ERROR"))

    


def load_images(path):
    """ Returns images from given location
      Images are loaded with BGRA color space
      Args:
        path (str) Path to images.
      Returns:
        image (tuple(ndarray)) Output array filled with loaded images.
      """
    images = dict()
    if os.path.isdir(path):
        for filename in os.listdir(path):
            img = cv2.imread(os.path.join(path, filename))
            if img is not None:
                images[filename] = img
    else:
        img = cv2.imread(path)
        if img is not None:
            filename = os.path.basename(path)
            images[filename] = img
    return images


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="MC - D2 - Barcode detection & recognition")
    parser.add_argument(
        "-i",
        "--input",
        help="input data path",
        default=_DEFAULT_INPUT_PATH, type=str)
    parser.add_argument(
        "-o",
        "--output",
        help="output data path",
        default=_DEFAULT_OUTPUT_PATH, type=str)
    parser.add_argument(
        "-v",
        "--verbose",
        help="increase output verbosity",
        action="store_true")
    args = parser.parse_args()
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    # Setup logging
    if args.verbose:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO
    logging.basicConfig(format="%(levelname)s: %(message)s", level=log_level)

    main(args)
