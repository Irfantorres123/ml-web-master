import numpy as np
import cv2
import os
from PIL import Image
from scipy import interpolate
from scipy.fft import fft, fftfreq, rfft, rfftfreq, irfft
import pandas as pd

RADIOGRAPH_PATH = "sunhl-1th-09-Jan-2017-218 C AP.jpg"


OUTPUT_ASPECT_RATIO = 0.6
HEIGHT_EXPANSION = 1.1

RESIZED_HEIGHT = 1024
MIN_SCAN_NECK = 0
MAX_SCAN_NECK = 0.4
ASPECT_RATIO = 0.5
CROP_BOTTOM_RATIO = 1


class Crop_radiograph_mask_roi():
    def __init__(self, radiograph):
        self.upper_boundary = None
        self.lower_boundary = None
        self.left_boundary = None
        self.right_boundary = None
        self.radiograph = radiograph

    def get_crop_boundaries(self, image):
        width, height = image.size
        top_of_img = height - 1
        break_outer_loop = False
        for h in range(height):
            for w in range(width):
                if image.getpixel((w, h)) > 20:
                    top_of_img = h
                    break_outer_loop = True
                    break
            if break_outer_loop is True:
                break

        self.upper_boundary = top_of_img

        center = image.size[0] / 2

        bottom_of_img = image.size[1] - 1
        break_outer_loop = False
        for h in range(height - 1, 0, -1):
            for w in range(width):
                if image.getpixel((w, h)) > 20:
                    bottom_of_img = h
                    break_outer_loop = True
                    break
            if break_outer_loop is True:
                break

        self.lower_boundary = bottom_of_img

        new_width = (self.lower_boundary - self.upper_boundary) * ASPECT_RATIO
        self.left_boundary = int(center - new_width / 2)
        self.right_boundary = int(center + new_width / 2)

    def crop_resize_save_images(self):
        _, height = self.radiograph.size
        self.get_crop_boundaries(self.radiograph)
        new_radiograph = self.radiograph.crop(
            (self.left_boundary, self.upper_boundary, self.right_boundary, self.lower_boundary))
        new_radiograph = new_radiograph.resize((int(RESIZED_HEIGHT * ASPECT_RATIO), RESIZED_HEIGHT),
                                               resample=Image.LANCZOS)
        return new_radiograph


def blur_edges(radiograph_cv, left_boundary, right_boundary, blur_cycle, blur_width, filter_size):
    blurred_left = cv2.blur(
        radiograph_cv[:, -1*left_boundary -
                      blur_width:-1*left_boundary+blur_width],
        ksize=(filter_size, filter_size))
    for j in range(blur_cycle-1):
        blurred_left = cv2.blur(blurred_left, ksize=(filter_size, filter_size))
    blurred_right = cv2.blur(
        radiograph_cv[:, right_boundary -
                      blur_width:right_boundary + blur_width],
        ksize=(filter_size, filter_size))
    for j in range(blur_cycle-1):
        blurred_right = cv2.blur(
            blurred_right, ksize=(filter_size, filter_size))
    radiograph_cv[:, -1*left_boundary-blur_width:-
                  1*left_boundary+blur_width] = blurred_left
    radiograph_cv[:, right_boundary -
                  blur_width:right_boundary+blur_width] = blurred_right
    return radiograph_cv


def preprocess_radiograph(radiograph):


    width, height = radiograph.size

    new_width = int(height * HEIGHT_EXPANSION * OUTPUT_ASPECT_RATIO)
    left_boundary = int((width - new_width) / 2)
    right_boundary = int((width + new_width) / 2)
    upper_boundary = -1 * int(height * (HEIGHT_EXPANSION - 1))
    new_radiograph = radiograph.crop(
        (left_boundary, upper_boundary, right_boundary, height - 1))
    # smooth the boundaries
    new_radiograph_cv = np.array(new_radiograph)

    new_radiograph_cv = blur_edges(new_radiograph_cv, left_boundary, right_boundary, blur_cycle=10,
                                   blur_width=80, filter_size=9)
    new_radiograph_cv = blur_edges(new_radiograph_cv, left_boundary, right_boundary, blur_cycle=10,
                                   blur_width=70, filter_size=9)
    new_radiograph_cv = blur_edges(new_radiograph_cv, left_boundary, right_boundary, blur_cycle=10,
                                   blur_width=60, filter_size=11)
    new_radiograph_cv = blur_edges(new_radiograph_cv, left_boundary, right_boundary, blur_cycle=10,
                                   blur_width=50, filter_size=15)
    new_radiograph_cv = blur_edges(new_radiograph_cv, left_boundary, right_boundary, blur_cycle=10,
                                   blur_width=30, filter_size=9)
    new_radiograph_cv = blur_edges(new_radiograph_cv, left_boundary, right_boundary, blur_cycle=10,
                                   blur_width=20, filter_size=5)

    radiograph_save_path = RADIOGRAPH_PATH.replace(".png", "_processed.png")

    # crop
    radiograph_pil = Image.fromarray(new_radiograph_cv)
    cropper = Crop_radiograph_mask_roi(radiograph_pil)
    resized_radiograph = cropper.crop_resize_save_images()

    # clahe
    resized_radiograph_cv = np.array(resized_radiograph)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 3))
    resized_radiograph_clahe = clahe.apply(resized_radiograph_cv)

    resized_radiograph_clahe_pil = Image.fromarray(resized_radiograph_clahe)

    # save images /csv
    #cv2.imwrite(radiograph_save_path, resized_radiograph_clahe)
    # cv2.imwrite(mask_save_path, mask_cv)
    return resized_radiograph_clahe_pil


if __name__ == "__main__":
    radiograph = Image.open(RADIOGRAPH_PATH).convert("L")
    processed_img = preprocess_radiograph(radiograph)
    processed_img.save(RADIOGRAPH_PATH.replace(".jpg", "_processed.png"))
    print("finished")
