import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras import layers
from keras.preprocessing import image
import matplotlib.image as mpimg
import numpy as np
import os
import logging
class RetinaModel:
    logger = logging.getLogger(__name__)
    class_names = ['CNV', 'DME', 'DRUSEN', 'NORMAL']
    # init method or constructor 
    def __init__(self, model_name, model_description, image_size):
        self.model_name = model_name
        self.model_description = model_description
        self.image_size = image_size
        self.logger.info("BEGIN ==> RetinaModel created with model_name = {}".format(model_name))
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(ROOT_DIR + '/static/model/')        
        self.tflite_model_file = os.path.join(path, model_name)
        with open(self.tflite_model_file, 'rb') as fid:
            self.tflite_model = fid.read()

        self.interpreter = tf.lite.Interpreter(model_content=self.tflite_model)
        self.interpreter.allocate_tensors()

        self.input_index = self.interpreter.get_input_details()[0]["index"]
        self.output_index = self.interpreter.get_output_details()[0]["index"]
        self.logger.info("END ==> RetinaModel created with model_name = {}".format(model_name))
    
    def read_image(self, image):
        return mpimg.imread(image)


    def format_image(self, image):
        # img = tf.image.resize(image[tf.newaxis, ...], [self.image_size, self.image_size]) / 255.0
        self.logger.info("format_image ==> original image shape is = {}".format(tf.shape(image)))
        img = np.stack((image,)*3, axis=-1)
        img = img / 255.0
        self.logger.info("format_image ==> image shape is = {}".format(tf.shape(img)))
        img1 = tf.image.resize(img, (self.image_size,self.image_size))
        new_img = np.expand_dims(img1, axis=0).astype(np.float32)
        self.logger.info("format_image ==> new_img shape is = {}".format(tf.shape(new_img)))
        return new_img

    # Sample Method 
    def  predict(self, img):
        self.logger.info("BEGIN ==> RetinaModel.predict created with model_name = {}".format(self.model_name))
        try:
            input_img = self.read_image(img)
            format_img = self.format_image(input_img)
            self.interpreter.set_tensor(self.input_index, format_img)
            self.interpreter.invoke()
            retVal = self.class_names[np.argmax(self.interpreter.get_tensor(self.output_index))]
        except Exception as e: 
             self.logger.error("RetinaModel predict exception {}".format(str(e)))
        self.logger.info('RetinaAI image class type = {}'.format(retVal))
        return retVal