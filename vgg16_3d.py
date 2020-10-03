import numpy as np
from keras import layers, models
from tensorflow import keras

from preprocessing import value_preprocessing
from train_framework import train, calculate_test_mae


def create_vgg16_3d(dense=False, batch_norm=True, weights=None, input_shape=(260, 100, 15, 1)):
    """
    Creates slightly modified VGG16 model, ported to 3d with trainable BatchNormalization
    """

    def create_conv(filter, kernel_size, name):
        def conv_wrapper(inp):
            x = layers.Conv3D(filter, kernel_size, padding='same', name=name)(inp)
            if batch_norm:
                x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            return x

        return conv_wrapper

    img_input = layers.Input(shape=input_shape, name='input')

    # block 1
    x = create_conv(8, (3, 3, 3), name='block1_conv1')(img_input)
    x = create_conv(8, (3, 3, 3), name='block1_conv2')(x)
    x = layers.MaxPooling3D((2, 2, 2), strides=(2, 2, 2), name='block1_pool')(x)

    # block 2
    x = create_conv(16, (3, 3, 3), name='block2_conv1')(x)
    x = create_conv(16, (3, 3, 3), name='block2_conv2')(x)
    x = layers.MaxPooling3D((2, 2, 2), strides=(2, 2, 2), name='block2_pool')(x)

    # block 3
    x = create_conv(32, (3, 3, 3), name='block3_conv1')(x)
    x = create_conv(32, (3, 3, 3), name='block3_conv2')(x)
    x = create_conv(32, (3, 3, 3), name='block3_conv3')(x)
    x = layers.MaxPooling3D((2, 2, 2), strides=(2, 2, 2), name='block3_pool')(x)

    # block 4
    x = create_conv(32, (3, 3, 3), name='block4_conv1')(x)
    x = create_conv(32, (3, 3, 3), name='block4_conv2')(x)
    x = create_conv(32, (3, 3, 3), name='block4_conv3')(x)
    # delete MaxPooling to allow for 5th block

    # block 5
    x = create_conv(64, (3, 3, 3), name='block5_conv1')(x)
    x = create_conv(64, (3, 3, 3), name='block5_conv2')(x)
    x = create_conv(64, (3, 3, 3), name='block5_conv3')(x)

    if dense:
        x = layers.Flatten()(x)
        x = layers.Dense(4096)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(2048)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(1)(x)
    else:
        x = layers.GlobalMaxPooling3D()(x)

    model = models.Model(img_input, x, name='vgg16_3d')

    if weights is not None:
        model.load_weights(weights, by_name=True)
    return model


def preprocessing(ct):
    """wrapper function to pass to generators"""
    return value_preprocessing(ct, False)


def output_reshape(ct):
    """wrapper function to pass to generators"""
    return np.moveaxis(ct, 1, -1)


name = 'vgg16_3d_bn_better0.1'
patch_size = (260, 100, 15)
model = create_vgg16_3d(dense=True)
optimizer = keras.optimizers.Adam(0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
train(model, optimizer, epochs=1000, batch_size=32, patch_size=patch_size, num_validation=32, name=name, loss='mae',
      preprocessing_func=preprocessing, output_reshape_func=output_reshape, training_generator_threads=6,
      training_sample_cache=16)
calculate_test_mae(model, optimizer, 'mae', 4, patch_size, preprocessing, output_reshape)
