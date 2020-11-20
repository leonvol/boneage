import numpy as np
from batchgenerators.dataloading import MultiThreadedAugmenter
from batchgenerators.dataloading.data_loader import DataLoader
from batchgenerators.transforms import Compose
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, GammaTransform
from batchgenerators.transforms.noise_transforms import GaussianBlurTransform, GaussianNoiseTransform
from batchgenerators.transforms.spatial_transforms import MirrorTransform
from batchgenerators.transforms.spatial_transforms import SpatialTransform_2
from tensorflow import keras

import util


def get_train_transform(patch_size):
    """
    data augmentation for training data, inspired by:
    https://github.com/MIC-DKFZ/batchgenerators/blob/master/batchgenerators/examples/brats2017/brats2017_dataloader_3D.py
    :param patch_size: shape of network's input
    :return list of transformations
    """

    train_transforms = []

    def rad(deg):
        return (-deg / 360 * 2 * np.pi, deg / 360 * 2 * np.pi)

    train_transforms.append(
        SpatialTransform_2(
            patch_size, (10, 10, 10),
            do_elastic_deform=True, deformation_scale=(0, 0.25),
            do_rotation=True,
            angle_z=rad(15),
            angle_x=(0, 0),
            angle_y=(0, 0),
            do_scale=True, scale=(0.75, 1.25),
            border_mode_data='constant', border_cval_data=0,
            border_mode_seg='constant', border_cval_seg=0,
            order_seg=1,
            random_crop=False,
            p_el_per_sample=0.2, p_rot_per_sample=0.2, p_scale_per_sample=0.2,
        )
    )

    train_transforms.append(MirrorTransform(axes=(0, 1)))

    train_transforms.append(BrightnessMultiplicativeTransform((0.7, 1.5), per_channel=True, p_per_sample=0.2))

    train_transforms.append(GammaTransform(gamma_range=(0.2, 1.0), invert_image=False, per_channel=False,
                                           p_per_sample=0.2))

    train_transforms.append(GaussianNoiseTransform(noise_variance=(0, 0.05), p_per_sample=0.2))

    train_transforms.append(GaussianBlurTransform(blur_sigma=(0.2, 1.0), different_sigma_per_channel=False,
                                                  p_per_channel=0.0, p_per_sample=0.2))

    return Compose(train_transforms)


def get_valid_transform(patch_size):
    """
    data augmentation for validation data
    inspired by:
    https://github.com/MIC-DKFZ/batchgenerators/blob/master/batchgenerators/examples/brats2017/brats2017_dataloader_3D.py
    :param patch_size: shape of network's input
    :return list of transformations
    """

    train_transforms = []

    train_transforms.append(
        SpatialTransform_2(
            patch_size, patch_size,
            do_elastic_deform=False, deformation_scale=(0, 0),
            do_rotation=False,
            angle_x=(0, 0),
            angle_y=(0, 0),
            angle_z=(0, 0),
            do_scale=False, scale=(1.0, 1.0),
            border_mode_data='constant', border_cval_data=0,
            border_mode_seg='constant', border_cval_seg=0,
            order_seg=1, order_data=3,
            random_crop=True,
            p_el_per_sample=0.1, p_rot_per_sample=0.1, p_scale_per_sample=0.1
        )
    )

    train_transforms.append(MirrorTransform(axes=(0, 1)))

    return Compose(train_transforms)


class CTBatchLoader(DataLoader):
    """
    Class to load and cache CT images
    Later used to augment data
    """

    def __init__(self,
                 data,
                 batch_size,
                 patch_size,
                 num_threads_in_multithreaded,
                 preprocess_func,
                 labels_path='data/ae_pseudonyms.csv',
                 seed_for_shuffle=1234,
                 return_incomplete=False,
                 shuffle=True,
                 infinite=True):
        """
        :param data: list of paths to files
        :param batch_size: batch size
        :param patch_size: shape of network's input
        :param num_threads_in_multithreaded: number of generators to run on different threads
        :param preprocess_func: callable to preprocess data
        :param labels_path: path to file containing labels for each path in the data parameter
        :param seed_for_shuffle: -
        :param return_incomplete: -
        :param shuffle: shuffle the data list after each batch
        :param infinite: repeat the dataset
        """

        super().__init__(data,
                         batch_size,
                         num_threads_in_multithreaded,
                         seed_for_shuffle,
                         return_incomplete,
                         shuffle,
                         infinite)
        self.patch_size = patch_size
        self.indices = list(range(len(data)))
        self.age_info = util.parse_labels_months(labels_path)
        self.preprocess_func = preprocess_func

    def generate_train_batch(self):
        """Loads data, preprocesses it and collect samples to a batch"""

        patients_indices = self.get_indices()
        patients_for_batch = [self._data[i] for i in patients_indices]

        data = np.zeros((self.batch_size, 1, *self.patch_size), dtype=np.short)
        labels = np.empty(self.batch_size, dtype=np.float32)

        # iterate over patients_for_batch and include them in the batch
        for i, j in enumerate(patients_for_batch):
            patient_data_ct = np.load(j).astype(np.short)

            data[i] = self.preprocess_func(patient_data_ct).astype(np.short)
            path = str(j).split('/')[-1].replace('.npy', '')
            labels[i] = float(self.age_info[path])

        return {'data': np.array(data), 'label': np.array(labels)}


class KerasGenerator(keras.utils.Sequence):
    """
    Class to make MIC-DKFZ batch generator output work for Keras models
    Allows reshaping the output to be more flexible for different network architectures
    """

    def __init__(self, batchgen, output_reshapefunc, n=12):
        """
        :param batchgen: object which implements __next__ and returns dict with labels 'data' and 'label', such as the
        CTBatchLoader or the MultithreadedAugmenter
        :param output_reshapefunc: callable to reshape output per sample
        :param n: number of steps per epoch
        """
        self.batch_generator = batchgen
        self.n = n
        self.output_reshape_func = output_reshapefunc

    def __len__(self):
        return self.n

    def __getitem__(self, item):
        return self.__next__()

    def __next__(self):
        batch = next(self.batch_generator)
        data = batch['data']
        label = batch['label']
        return self.output_reshape_func(data), label


def get_generators(patch_size, batch_size, preprocess_func, output_reshape_func, num_validation, train_processes,
                   train_cache, dirs):
    """
    Creates augmented batch loaders and generators for Keras for training and validation
    :param patch_size: input of network without batch_size dimension
    :param batch_size:
    :param preprocess_func: callable to preprocess data per sample
    :param output_reshape_func: callable to reshape preprocessed and augmented data per sample
    :param num_validation: number of samples to validate on
    :param train_processes: number of threads to load, preprocess and augment data
    :param  train_cache: number of augmented samples to cache
    """

    # dirs = util.get_data_list(train_data_dir)
    labels = util.parse_labels_months()
    train_paths, validation_paths = util.train_validation_split(dirs, labels)
    # generate train batch loader
    train_data_loader = CTBatchLoader(train_paths, batch_size, patch_size, num_threads_in_multithreaded=1,
                                      preprocess_func=preprocess_func)

    train_transforms = get_train_transform(patch_size)
    train_data_generator = MultiThreadedAugmenter(train_data_loader, train_transforms, num_processes=train_processes,
                                                  num_cached_per_queue=train_cache, seeds=None, pin_memory=False)

    # wrapper to be compatible with keras
    train_generator_keras = KerasGenerator(train_data_generator, output_reshapefunc=output_reshape_func)

    # generate validation batch loader
    valid_data_loader = CTBatchLoader(validation_paths, num_validation, patch_size,
                                      num_threads_in_multithreaded=1, preprocess_func=preprocess_func)
    valid_transforms = get_valid_transform(patch_size)
    valid_data_generator = MultiThreadedAugmenter(valid_data_loader, valid_transforms, num_processes=1,
                                                  num_cached_per_queue=1, seeds=None, pin_memory=False)
    # wrapper to be compatible with keras
    valid_generator_keras = KerasGenerator(valid_data_generator, output_reshape_func, 1)

    return train_generator_keras, valid_generator_keras


def get_test_generator(patch_size, batch_size, preprocess_func, output_reshape_func, test_paths):
    """
    Creates un-augmented data generator/loader for testing data
    Especially useful if testing data does not fit into memory
    :param patch_size: input of network without batch_size dimension
    :param batch_size:
    :param preprocess_func: callable to preprocess data per sample
    :param output_reshape_func: callable to reshape preprocessed and augmented data per sample
    """

    # generate train batch loader
    test_data_loader = CTBatchLoader(test_paths, batch_size, patch_size, num_threads_in_multithreaded=1,
                                     preprocess_func=preprocess_func, infinite=False)

    # wrapper to be compatible with keras
    return KerasGenerator(test_data_loader, output_reshapefunc=output_reshape_func,
                          n=int(len(test_data_loader.indices) / batch_size))
