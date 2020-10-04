from tensorflow import keras

import util


def predict(dicom_dir, nrrd_path, preprocess_func, output_shape_func, model_path='models/vgg16_bn_final',
            dims=(130, 100, 15)):
    """Function to make loading, cropping, preprocessing and in general predicting of a single not yet preprocessed CT image easier"""

    dicom = util.load_dicom_as_numpy(dicom_dir)
    mask_nonzero = util.load_nrrd(nrrd_path)
    cropped = util.crop_dicom(dicom, mask_nonzero, dims)
    try:
        model = keras.models.load_model(model_path)
        prediction = model.predict(cropped)
        print('The predicted age is {} months'.format(prediction))
    except Exception as e:
        print('Catched exception while loading/predicting: ' + str(e))

# example usage:
# from vgg16_3d import preprocessing, output_reshape
# ct = 'data/sample/'
# nrrd = 'data/sample.nrrd'
# predict(ct, nrrd, preprocessing, output_reshape)
