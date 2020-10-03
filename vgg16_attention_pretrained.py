import numpy as np
from tensorflow import keras

from train_framework import train, calculate_test_mae


def create_vgg16_pretrained(shape=(384, 384, 3), trainable=False):
    """
    Slightly modified version of the kaggle following model with pre-trained weights
    https://www.kaggle.com/kmader/attention-on-pretrained-vgg16-for-bone-age
    """

    in_lay = keras.Input(shape)
    base_pretrained_model = keras.applications.VGG16(input_shape=shape, include_top=False, weights='imagenet')
    base_pretrained_model.trainable = False

    pt_depth = 512
    pt_features = base_pretrained_model(in_lay)
    bn_features = keras.layers.BatchNormalization()(pt_features)

    attn_layer = keras.layers.Conv2D(64, kernel_size=(1, 1), padding='same', activation='relu')(bn_features)
    attn_layer = keras.layers.Conv2D(16, kernel_size=(1, 1), padding='same', activation='relu')(attn_layer)
    attn_layer = keras.layers.LocallyConnected2D(1,
                                                 kernel_size=(1, 1),
                                                 padding='valid',
                                                 activation='sigmoid')(attn_layer)

    up_c2_w = np.ones((1, 1, 1, pt_depth))
    up_c2 = keras.layers.Conv2D(pt_depth, kernel_size=(1, 1), padding='same',
                                activation='linear', use_bias=False, weights=[up_c2_w])
    up_c2.trainable = False
    attn_layer = up_c2(attn_layer)

    mask_features = keras.layers.multiply([attn_layer, bn_features])
    gap_features = keras.layers.GlobalAveragePooling2D()(mask_features)
    gap_mask = keras.layers.GlobalAveragePooling2D()(attn_layer)

    gap = keras.layers.Lambda(lambda x: x[0] / x[1], name='RescaleGAP')([gap_features, gap_mask])
    gap_dr = keras.layers.Dropout(0.5)(gap)
    dr_steps = keras.layers.Dropout(0.25)(keras.layers.Dense(1024, activation='elu')(gap_dr))
    out_layer = keras.layers.Dense(1, activation='linear')(dr_steps)  # linear is what 16bit did

    pretrained_model = keras.Model(inputs=[in_lay], outputs=[out_layer])
    pretrained_model.trainable = trainable

    try:
        pretrained_model.load_weights('models/pretrained/modelvgg16-bone.hdf5')
    except Exception as e:
        print("Could not load Kaggle base model weights: " + str(e))

    return pretrained_model


def create_model(shape=(384, 348, 3), n_slices=15):
    """
    Create network which uses outputs of a pretrained keras model from kaggle for each CT layer
    """
    inputs = keras.Input(shape=(n_slices,) + shape)
    base_model = create_vgg16_pretrained(shape)

    vggs = keras.layers.TimeDistributed(base_model)(inputs)
    dense0 = keras.layers.Dense(2048, activation='relu')(vggs)
    batch_norm0 = keras.layers.BatchNormalization()(dense0)
    dense1 = keras.layers.Dense(2048, activation='relu')(batch_norm0)
    batch_norm1 = keras.layers.BatchNormalization()(dense1)
    dense3 = keras.layers.Dense(1024, activation='relu')(batch_norm1)
    batch_norm2 = keras.layers.BatchNormalization()(dense3)
    output = keras.layers.Dense(1)(batch_norm2)
    model = keras.Model(inputs=[inputs], outputs=[output])

    return model


single_layer_shape = (384, 384)
channels = 3
layers = 15


def preprocess(ct):
    """preprocesses each loaded CT image"""
    from preprocessing import value_preprocessing
    preprocessed = value_preprocessing(ct).astype(np.short)
    img = np.swapaxes(preprocessed, 0, 2)
    end = np.zeros((layers,) + single_layer_shape, dtype=np.short)
    for i in range(end.shape[0]):
        y = (end.shape[1] - img.shape[1]) // 2
        x = (end.shape[2] - img.shape[2]) // 2
        end[i, y:y + img.shape[1], x:x + img.shape[2]] = img[i]
    return end.swapaxes(0, 2).astype(np.short)


def reshape_output(batch):
    """reshapes batch output to fit input to NN"""
    batch_size, _, height, width, layers = batch.shape
    reshaped = np.zeros((batch_size, layers, height, width, channels))
    for sample in range(batch_size):
        sample_ = batch[sample][0]
        stacked = np.array([sample_, sample_, sample_]).swapaxes(0, 2)
        stacked = np.moveaxis(stacked, -1, 0)
        reshaped[sample, :, :, :] = stacked
    return reshaped.astype(np.short)


name = 'vgg16_pretrained'
model = create_model(single_layer_shape + (channels,), layers)
optimizer = keras.optimizers.Adam(0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
train(model, optimizer, epochs=1000, batch_size=2, patch_size=single_layer_shape + (layers,), num_validation=2,
      name=name, loss='mae', preprocessing_func=preprocess, output_reshape_func=reshape_output,
      training_generator_threads=10, training_sample_cache=10)
calculate_test_mae(model, optimizer, loss='mae', batch_size=2, patch_size=single_layer_shape + (layers,),
                   preprocessing_func=preprocess, output_reshape_func=reshape_output)
