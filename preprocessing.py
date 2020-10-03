import numpy as np
from scipy.ndimage import sobel
from skimage import feature
from tensorflow import keras


def value_preprocessing(ct, stack=False):
    """
    preprocesses input ct image, applies:
    1. HU thresholding
    2. vgg16 preprocessing
    3. vgg16 like normalizing for non masked values

    :param ct: numpy array of data
    :param stack: stack same image 3 times to simulate rgb format
    :returns preprocessed data
    """
    height, width, layers = ct.shape
    if stack:
        preprocessed = np.zeros((height, width, layers, 3))
    else:
        preprocessed = np.zeros_like(ct)

    for layer_index in range(ct.shape[-1]):
        tmp_layer = ct[:, :, layer_index]
        # apply hu thresholding
        bone_thresholded = apply_hounsfield_thresholding(tmp_layer, (670, 2000))

        # apply recommended vgg16 preprocessing --> moving values from ~-130 to ~2000
        vgg_preprocessed = keras.applications.vgg16.preprocess_input(bone_thresholded)

        # applying custom normaliziation to move to ~-130 to ~150
        # value 0.075: is roughly 150 / 2000
        vgg_preprocessed[np.logical_and(vgg_preprocessed < 650, vgg_preprocessed > 0)] *= 0.075
        vgg_preprocessed[vgg_preprocessed >= 700] *= 0.075

        # triple stacking arrays
        if stack:
            stacked = np.array([vgg_preprocessed, vgg_preprocessed, vgg_preprocessed])
            stacked = np.moveaxis(stacked, 0, 2)
            preprocessed[:, :, layer_index, :] = stacked
        else:
            preprocessed[:, :, layer_index] = vgg_preprocessed

    return preprocessed


def apply_sobel(data_):
    """Looks for corners with sobel edge detection"""
    return sobel(data_)


def apply_hounsfield_thresholding(data_, threshold: tuple = (200, 600)):
    """Looks for bones by thresholding image"""
    mask = np.ma.masked_inside(data_, threshold[0], threshold[1], ).mask
    thresholded = np.zeros_like(data_)
    thresholded[mask] = data_[mask]
    return thresholded


def apply_canny(data_, sigma=3):
    """Detect edges of clavicle with canny"""
    edges = np.zeros_like(data_)
    for i in range(data_.shape[-1]):
        img = data_[:, :, i]
        tmp_edges = feature.canny(img, sigma=sigma)
        edges[:, :, i] = tmp_edges
    return edges


def normalize_houndsfield(data_):
    """Normalizes houndsfield values ranging from -1024 to ~+4000 to (0, 1)"""
    cpy = data_ + 1024
    cpy /= 3000
    return cpy
