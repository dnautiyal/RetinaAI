import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras import layers
from tensorflow.keras.preprocessing import image
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
    
    def get_model_name(self):
        return self.model_name
    
    def get_model_description(self):
        return self.model_description
    
    # method to predict from 'img' object which is the object received from webpage's multipart upload
    def  predict(self, input_image_filename, formatted_img):
        self.logger.info("BEGIN ==> RetinaModel.predict created with model_name = {}".format(self.model_name))
        ret_val = {}
        try:
            ret_val['Model'] = self.get_model_description()
            self.interpreter.set_tensor(self.input_index, formatted_img)
            self.interpreter.invoke()
            preds = self.interpreter.get_tensor(self.output_index)
            pred_index = np.argmax(preds)
            prediction_score = preds[0][pred_index]
            class_name = self.class_names[pred_index]
            ret_val['Prediction-Class'] = class_name
            ret_val['Prediction-Score'] = "{:.2%}".format(prediction_score)
        except Exception as e: 
            self.logger.error("RetinaModel predict exception {}".format(str(e)))
            class_name = "ERROR"
            prediction_score = 0
            ret_val['Prediction-Class'] = class_name
            ret_val['Prediction-Score'] = "{:.2%}".format(prediction_score)
            return ret_val            
        self.logger.info('RetinaAI Image ({}) class type = {}, prediction_score ={}, predictions ={}'.format(input_image_filename, class_name, prediction_score, np.array_str(preds,precision = 4)))
        return ret_val

    
