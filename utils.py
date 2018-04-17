import math
import os
import random
import re
import string

import numpy as np
import scipy.misc
import tensorlayer as tl

""" The functions here will be merged into TensorLayer after finishing this project.
"""


def load_and_assign_npz(sess=None, name="", model=None):
    assert model is not None
    assert sess is not None
    if not os.path.exists(name):
        print("[!] Loading {} model failed!".format(name))
        return False
    else:
        params = tl.files.load_npz(name=name)
        tl.files.assign_params(sess, params, model)
        print("[*] Loading {} model SUCCESS!".format(name))
        return model


# files
def load_folder_list(path=""):
    """Return a folder list in a folder by given a folder path.

    Parameters
    ----------
    path : a string or None
        A folder path.
    """
    return [os.path.join(path, o) for o in os.listdir(path) if os.path.isdir(os.path.join(path, o))]


# utils
def print_dict(dictionary={}):
    """Print all keys and items in a dictionary.
    """
    for key, value in dictionary.items():
        print("key: %s  value: %s" % (str(key), str(value)))


# prepro ?
def get_random_int(min=0, max=10, number=5):
    """Return a list of random integer by the given range and quantity.

    Examples
    ---------
    >>> r = get_random_int(min=0, max=10, number=5)
    ... [10, 2, 3, 3, 7]
    """
    return [random.randint(min, max) for p in range(0, number)]


def preprocess_caption(line):
    prep_line = re.sub('[%s]' % re.escape(string.punctuation), ' ', line.rstrip())
    prep_line = prep_line.replace('-', ' ')
    return prep_line


## Save images
def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j * h:j * h + h, i * w:i * w + w, :] = image
    return img


def imsave(images, size, path):
    return scipy.misc.imsave(path, merge(images, size))


def save_images(images, size, image_path):
    return imsave(images, size, image_path)


from tensorlayer.prepro import *


def prepro_img(data, img_size=64):
    # rescale [0, 255] --> (-1, 1), random flip, crop, rotate
    #   paper 5.1: During mini-batch selection for training we randomly pick
    #   an image view (e.g. crop, flip) of the image and one of the captions
    # flip, rotate, crop, resize : https://github.com/reedscot/icml2016/blob/master/data/donkey_folder_coco.lua
    # flip : https://github.com/paarthneekhara/text-to-image/blob/master/Utils/image_processing.py

    x, c = data

    if np.random.choice([True, False]):
        x = flip_axis(x, axis=1)
        c = [img_size - c[0], c[1]]

    r = np.random.choice(32) - 16
    x = rotation(x, rg=r, fill_mode='nearest')
    c = rotate([img_size / 2, img_size / 2], c, r)

    x = imresize(x, size=[img_size + 15, img_size + 15], interp='bilinear')
    s = (img_size + 15) / img_size
    c = [c[0] * s, c[1] * s]

    x, _, new_coords = obj_box_crop(x, coords=[[c[0], c[1], 20, 20]], classes=[0], wrg=img_size, hrg=img_size, is_random=True)
    c = [new_coords[0][0], new_coords[0][1]]

    x = x / (255. / 2.)
    x = x - 1.

    return x, c


def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in degrees.
    """
    angle = math.radians(angle)
    ox, oy = origin
    px, py = point

    cos = math.cos(angle)
    sin = math.sin(angle)

    qx = ox + cos * (px - ox) - sin * (py - oy)
    qy = oy + sin * (px - ox) + cos * (py - oy)
    return [qx, qy]


def combine_and_save_image_sets(image_sets, directory):
    for i in range(len(image_sets[0])):
        combined_image = []
        for set_no in range(len(image_sets)):
            combined_image.append(image_sets[set_no][i])
            combined_image.append(np.zeros((image_sets[set_no][i].shape[0], 5, 3)))
        combined_image = np.concatenate(combined_image, axis=1)

        scipy.misc.imsave(os.path.join(directory, 'combined_{}.jpg'.format(i)), combined_image)


def get_center(coords):
    x = coords[0]
    y = coords[1]
    w = coords[2]
    h = coords[3]
    return [x + w / 2, y + h / 2]
