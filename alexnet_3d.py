import keras
import numpy as np
from keras.layers import Dense, Activation, Dropout, Flatten, Conv3D, MaxPooling3D, BatchNormalization
from tensorflow import keras

from preprocessing import value_preprocessing
from train_framework import train, calculate_test_mae


def create_alexnet_3d(batchnorm=False, dropout=False, weights=None, input_shape=(260, 100, 15, 1)):
    """
    AlexNet in 3D.
    """

    img_input = keras.Input(shape=input_shape)

    # 1st Convolutional Layer
    x = Conv3D(filters=6, kernel_size=5, strides=1, input_shape=input_shape, padding='same')(img_input)
    if batchnorm: x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = MaxPooling3D(pool_size=3, strides=2, padding='same')(x)

    # 2nd Convolutional Layer
    x = Conv3D(filters=16, kernel_size=4, strides=1, padding='same')(x)
    if batchnorm: x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = MaxPooling3D(pool_size=3, strides=2, padding='same')(x)

    # 3rd Convolutional Layer
    x = Conv3D(filters=24, kernel_size=3, strides=1, padding='same')(x)
    if batchnorm: x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # 4th Convolutional Layer
    x = Conv3D(filters=24, kernel_size=3, strides=1)(x)
    if batchnorm: x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # delete 5th Conv Layer and Pooling to allow for input of the chosen size

    x = Flatten()(x)

    x = Dense(4096, activation='relu')(x)
    x = BatchNormalization()(x)
    if dropout: x = Dropout(0.3)(x)
    x = Dense(2048, activation='relu')(x)
    x = BatchNormalization()(x)
    if dropout: x = Dropout(0.3)(x)
    x = Dense(1024, activation='relu')(x)
    x = BatchNormalization()(x)
    if dropout: x = Dropout(0.3)(x)

    output = Dense(1)(x)

    model = keras.Model(inputs=img_input, outputs=[output])

    # load weights
    if weights is not None:
        model.load_weights(weights, by_name=True)

    return model


def preprocessing(ct):
    """wrapper function to pass to generators"""
    return value_preprocessing(ct, False)


def output_reshape(ct):
    """wrapper function to pass to generators"""
    return np.moveaxis(ct, 1, -1)


name = 'alexnet_3d'
patch_size = (260, 100, 15)
model = create_alexnet_3d(True, True)
optimizer = keras.optimizers.Adam(0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
train(model, optimizer, epochs=1000, batch_size=32, patch_size=patch_size, num_validation=32, name=name, loss='mae',
      preprocessing_func=preprocessing, output_reshape_func=output_reshape, training_generator_threads=6,
      training_sample_cache=16)
calculate_test_mae(model, optimizer, 'mae', 4, patch_size, preprocessing, output_reshape)