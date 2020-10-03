import os
import time

import dicom_numpy
import nrrd
import numpy as np
import pydicom


def load_dicom_as_numpy(dicom_dir: str) -> np.ndarray:
    """
    Method to load ct image in numpy, slightly adapted form https://dicom-numpy.readthedocs.io/en/latest/
    :param dicom_dir: directory where dicom data is located
    :return: numpy array of ct data
    """
    datasets = [pydicom.read_file(dicom_dir + f) for f in os.listdir(dicom_dir)]
    try:
        voxel_ndarray, ijk_to_xyz = dicom_numpy.combine_slices(datasets)
    except dicom_numpy.DicomImportException:
        raise
    return voxel_ndarray


def load_nrrd(mask_path: str) -> tuple:
    """
    Loads nrrd mask and returns non zero position
    :param mask_path: path to load mask from
    :return: nonzero position
    """
    try:
        mask = nrrd.read(mask_path)[0]
        nonzero = np.nonzero(mask)
        return nonzero[0][0], nonzero[1][0], nonzero[2][0]
    except FileNotFoundError:
        raise


def crop(ct, mask_pos: tuple, dims: tuple):
    """
    Method to crop full ct image
    :param ct: ct image as numpy array 
    :param mask_pos: position to crop from
    :param dims: dimensions to crop from mask in format (x,y,z), x gets cropped in both positive and negative direction
    :return: cropped
    """
    x, y, z = mask_pos

    x_dim, y_dim, z_dim = dims
    y_basic_offset = 40

    return ct[x - x_dim: x + x_dim, y - y_basic_offset:y - y_basic_offset + y_dim, z:z + z_dim]


def preprocess_dir(data_dir: str, save_dir: str, ignored_path: str, dims: tuple):
    """
    Method to crop the whole data directory
    :param data_dir: data directory
    :param save_dir: directory to save the cropped data to 
    :param ignored_path: path to file containing ignored directories
    :param dims: determines dimensions of cropped image dimensions of cropped image in format (x, y, z) which results in
    """
    t_start = time.time()

    # parse dir names of ignored dirs
    file_false_dirs = open(ignored_path)
    lines = file_false_dirs.readlines()
    ignored_dirs = list()
    for line in lines:
        dir_name = line.split(" ")[0].replace("\n", "").replace(":", "")
        ignored_dirs.append(dir_name)

    # loop over data
    for filename in os.listdir(data_dir):
        if os.path.isdir(data_dir + filename):
            if filename in ignored_dirs:
                print("Ignoring {}".format(filename))
            else:
                print("Preprocessing {}".format(data_dir + filename))
                # there can be more than one directory and according .nrrd file
                for subfilename in os.listdir(data_dir + filename):
                    path = data_dir + filename + "/" + subfilename
                    if os.path.isdir(path):
                        print("Preprocessing subdir {}".format(path))
                        try:
                            dicom = load_dicom_as_numpy(path + "/")
                            mask = load_nrrd(path + ".nrrd")
                            cropped = crop(dicom, mask, dims)
                            np.save(save_dir + subfilename, cropped)
                            print("Finished cropping {}".format(path))
                        except dicom_numpy.DicomImportException:
                            print("DicomImportException at {}".format(path))
                        except FileNotFoundError:
                            print("No .nnrd file found for {}".format(path))
                print("Finished cropping  {}".format(data_dir + filename))
    print("Finished preprocessing in {} min".format((time.time() - t_start) / 60))
