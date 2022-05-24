import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from .model import RetinaModel
import logging
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing import image
# logging.basicConfig(filename='/opt/python/log/retinaai.log', level=logging.DEBUG)
logger = logging.getLogger(__name__)
# model_1 = RetinaModel('Retinal_OCT_INCEPTIONV3-04062022-00-33.tflite', 'INCEPTIONV3', 256)
IMAGE_SIZE = 256

models = [RetinaModel('tflite_dense121_best', 'DENSENET121', IMAGE_SIZE),
            RetinaModel('converted_model_mobilenetv2_20May.tflite', 'MOBILENETV2', IMAGE_SIZE),
            RetinaModel('Retinal_OCT_VGG16-05212022-08-05.tflite', 'VGG16', IMAGE_SIZE)]
def get_prediction(img_from_upload):
    """Write a Function to Predict the Class Name

    Args:
        img [jpg]: image file

    Returns:
        [str]: Prediction
    """
    # saved_image_filepath = save_image_to_file(img_from_upload)
    # logger.info("saved file to {}" .format(saved_image_filepath))
    logger.info("calling model.predict method")
    # formatted_image = preprocess_image_tf(saved_image_filepath)
    formatted_image = preprocess_image(img_from_upload)
    data = []
    for model in models:
        pred = model.predict(img_from_upload.filename, formatted_image)
        data.append(pred)
    df = pd.DataFrame(data)
    ret_val = df.to_html(justify='center', index = False, escape=False, classes='pred-table')
    logger.info(df.head(len(models))) 
    # logger.info(ret_val)
    return ret_val

def save_image_to_file(img, save_image_filename='/static/images/input.jpeg'):
    """Plot the input image

    Args:
        img [jpg]: image file
    """
    read_img = mpimg.imread(img)
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(ROOT_DIR + save_image_filename)
    plt.imsave(path, read_img)
    return path

def create_output_image(img):
    """Plot the input image

    Args:
        img [jpg]: image file
    """
    save_image_to_file(img, '/static/images/output.png')

def preprocess_image_tf(image_path):
    new_img = image.load_img(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    logger.info("preprocess_image_tf ==> new_img shape is = {}".format(tf.shape(new_img)))
    img = image.img_to_array(new_img)
    img = np.expand_dims(img, axis=0)
    img = img/255
    img = np.array(img, dtype=np.float32)
    logger.info("format_image ==> img shape is = {}".format(tf.shape(img)))
    return img

def preprocess_image(image_from_upload):
    image = mpimg.imread(image_from_upload)
    # img = tf.image.resize(image[tf.newaxis, ...], [self.image_size, self.image_size]) / 255.0
    logger.info("format_image ==> original image shape is = {}".format(tf.shape(image)))
    img = np.stack((image,)*3, axis=-1)
    img = img / 255.0
    logger.info("format_image ==> image shape is = {}".format(tf.shape(img)))
    img1 = tf.image.resize(img, (IMAGE_SIZE,IMAGE_SIZE))
    new_img = np.expand_dims(img1, axis=0).astype(np.float32)
    logger.info("format_image ==> new_img shape is = {}".format(tf.shape(new_img)))
    return new_img