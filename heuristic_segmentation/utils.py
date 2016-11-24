# -*- coding: utf-8 -*-
import os
from os.path import isfile, join
from os import listdir
import sys
import linecache
import numpy as np
import cv2

def print_exception():
    exc_type, exc_obj, tb = sys.exc_info()
    f = tb.tb_frame
    lineno = tb.tb_lineno
    filename = f.f_code.co_filename
    linecache.checkcache(filename)
    line = linecache.getline(filename, lineno, f.f_globals)
    return 'EXCEPTION IN ({}, LINE {} "{}"): {}'.format(filename, lineno, line.strip(), exc_obj)

def get_client_ip(request):
    x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
    if x_forwarded_for:
        ip = x_forwarded_for.split(',')[0]
    else:
        ip = request.META.get('REMOTE_ADDR')
    return ip

def test_create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return dir_path

def fit_image_into_frame(img, frame_size=(200, 200, 3), random_fill=True, fill_color=[255, 255, 255], mode='crop'):
    if mode == 'fit':
        Y1, X1, _ = frame_size
        if random_fill:
            image_frame = np.asarray(np.random.randint(0, high=255, size=frame_size), dtype='uint8')
        else:
            image_frame = np.ones(frame_size, dtype='uint8')
            image_frame = image_frame*fill_color
            image_frame = np.asarray(image_frame, dtype='uint8')

        Y2, X2 = img.shape[0], img.shape[1]

        # scale up/down images with respect to largest size {width or height}
        if X2 > Y2:
            X_new = X1
            Y_new = int(round(float(Y2*X_new)/float(X2)))
        else:
            Y_new = Y1
            X_new = int(round(float(X2*Y_new)/float(Y2)))

        img = cv2.resize(img, (X_new, Y_new))

        # if image other side is larger then frame the scale that size.
        Y2, X2 = img.shape[0], img.shape[1]
        if Y2>Y1:
            Y_new = Y1
            X_new = int(round(float(X2*Y_new)/float(Y2)))
            img = cv2.resize(img, (X_new, Y_new))
        if X2>X1:
            X_new = X1
            Y_new = int(round(float(Y2*X_new)/float(X2)))
            img = cv2.resize(img, (X_new, Y_new))

        # convert from (w, h) to (w, h, 1), if it has single/1 channel
        if frame_size[2] == 1:
            img = img[..., None]

        X_space_center = ((X1 - X_new)/2)
        Y_space_center = ((Y1 - Y_new)/2)

        image_frame[Y_space_center: Y_space_center+Y_new, X_space_center: X_space_center+X_new] = img

    elif mode == 'crop':
        X1, Y1, _ = frame_size
        image_frame = np.zeros(frame_size, dtype='uint8')

        X2, Y2 = img.shape[1], img.shape[0]

        #increase the size of smaller length (width or hegiht)
        if X2 > Y2:
            Y_new = Y1
            X_new = int(round(float(X2*Y_new)/float(Y2)))
        else:
            X_new = X1
            Y_new = int(round(float(Y2*X_new)/float(X2)))

        img = cv2.resize(img, (X_new, Y_new))

        
        X_space_clip = (X_new - X1)/2
        Y_space_clip = (Y_new - Y1)/2

        #trim image both top, down, left and right
        if X_space_clip == 0 and Y_space_clip != 0:
            img = img[Y_space_clip:-Y_space_clip, :]
        elif Y_space_clip == 0 and X_space_clip != 0:
            img = img[:, X_space_clip:-X_space_clip]

        if img.shape[0] != X1:
            img = img[1:, :]
        if img.shape[1] != Y1:
            img = img[:, 1:]

        image_frame[: , :] = img
    return image_frame


def get_immediate_subdirectories(dir):
    """
        this function return the immediate subdirectory list
        eg:
            return ['subdirectory1',subdirectory2',...]
    """
    return [name for name in os.listdir(dir) if os.path.isdir(os.path.join(dir, name))]

def get_files(dir_path):
    """
        gets list of files in specified directory
        dir_path (String):
            directory path where we get all files
    """
    onlyfiles = [ f for f in listdir(dir_path) if isfile(join(dir_path, f)) ]
    return onlyfiles