import tensorflow as tf
from keras import Model
from keras.optimizers import Optimizer
from tensorflow import keras

from batch_loader import get_generators, get_test_generator
from clr_callback import CyclicLR


def train(model: Model, optimizer: Optimizer, epochs: int, batch_size: int, patch_size: tuple, num_validation: int,
          name: str, loss: str, preprocessing_func, output_reshape_func, training_generator_threads,
          training_sample_cache, train_paths, load_path=None):
    """
    Universal method to make training of different networks with different input_sizes easier and to minimize code duplication
    :param model: tf.keras model instance
    :param optimizer: tf.keras.optimizers.Optimizer
    :param epochs: number of epochs to train the model
    :param batch_size: batch size
    :param patch_size: network input shape
    :param num_validation: number of samples to validate on
    :param name: unique name to identify model
    :param loss: string like 'mae' or custom loss function
    :param preprocessing_func: callable to preprocess loaded data
    :param output_reshape_func: callable to reshape output to fit into the network
    :param training_generator_threads: number of threads the generator should run on
    :param training_sample_cache: number of samples to cache
    :param load_path: path to load model from
    :return: training history
    """

    model.compile(optimizer, loss=loss, metrics=['mse'])
    if load_path is not None:
        model.load_weights(load_path)

    save_path = 'models/{}/best'.format(name)
    model.save_weights(save_path)
    checkpointing = keras.callbacks.ModelCheckpoint(save_path, monitor='val_loss', verbose=1,
                                                    save_best_only=True, mode='min', save_weights_only=False)

    clr = CyclicLR(base_lr=0.001, max_lr=0.01, step_size=2000.)
    tensorboard = keras.callbacks.TensorBoard(log_dir='./graphs/{}/graph'.format(name), histogram_freq=0,
                                              write_graph=True, write_images=True)
    es = keras.callbacks.EarlyStopping(monitor='val_loss', mode='auto', patience=200, verbose=1)
    callbacks = [checkpointing, clr, tensorboard, es]

    train_generator, validation_generator = get_generators(patch_size, batch_size, preprocessing_func,
                                                           output_reshape_func, num_validation,
                                                           training_generator_threads, training_sample_cache,
                                                           train_paths)

    hist = model.fit(train_generator, epochs=epochs, callbacks=callbacks,
                     validation_data=validation_generator,
                     max_queue_size=0)
    model.save_weights('models/{}/final'.format(name))
    return hist


def calculate_test_mae(model: Model, optimizer, loss, batch_size, patch_size, preprocessing_func, output_reshape_func, paths):
    """
    Calculates mae on test set
    Creates data generator to load testing data on demand
    Useful if test set cannot fit into memory
    :param model: tf.keras model instance
    :param optimizer: tf.keras.optimizers.Optimizer
    :param batch_size: batch size
    :param patch_size: network input shape
    :param loss: string like 'mae' or custom loss function
    :param preprocessing_func: callable to preprocess loaded data
    :param output_reshape_func: callable to reshape output to fit into the network
    """
    model.compile(optimizer, loss)
    test_generator = get_test_generator(patch_size, batch_size, preprocessing_func, output_reshape_func, paths)
    return model.evaluate(test_generator, max_queue_size=0, steps=int(56 / batch_size), verbose=1)
