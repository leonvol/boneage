import numpy as np
from keras import layers, models
from tensorflow import keras
from util import get_data_list
from preprocessing import value_preprocessing
from train_framework import train, calculate_test_mae
from sklearn.model_selection import KFold


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


def _assemble_paths(paths, indices):
    return [paths[i] for i in indices]


name = 'vgg16_3d'
patch_size = (260, 100, 15)
model = create_vgg16_3d(dense=True)
start_weights = model.get_weights()
optimizer = keras.optimizers.Adam(0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

# cross validation
# final test maes for each cross validation iteration
test_maes = []
# training paths for each cross validation iteration
train_paths_ = []
# test paths for each cross validation iteration
test_paths_ = []
data_path = 'data/data/'
data_paths = get_data_list(data_path)

kf = KFold(8, True)
count = 0
for train_paths, test_paths in kf.split(data_paths):
    model.set_weights(start_weights)
    print('Start training ' + str(count))
    # convert from indices to actual paths, TODO change to use indices
    train_paths = _assemble_paths(data_paths, train_paths)
    test_paths = _assemble_paths(data_paths, test_paths)
    train_paths_.append(train_paths)
    test_paths_.append(test_paths)
    train(model, optimizer, epochs=500, batch_size=32, patch_size=patch_size, num_validation=32, name=name, loss='mae',
          preprocessing_func=preprocessing, output_reshape_func=output_reshape, training_generator_threads=6,
          training_sample_cache=16, train_paths=train_paths)
    mae = calculate_test_mae(model, optimizer, 'mae', 4, patch_size, preprocessing, output_reshape, test_paths)
    test_maes.append(mae)
    print('Finished testing ' + str(count) + ' with a mae of: ' + str(mae))
    model.save_weights('models/vgg16_3d_kf{}'.format(count))
    count += 1
print('test maes', test_maes)
print('training paths', train_paths_)
print('testing paths', test_paths_)
