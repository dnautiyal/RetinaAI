import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from .model import RetinaModel
import logging
# logging.basicConfig(filename='/opt/python/log/retinaai.log', level=logging.DEBUG)
logger = logging.getLogger(__name__)
model_1 = RetinaModel('Retinal_OCT_INCEPTIONV3-04062022-00-33.tflite', 'INCEPTIONV3', 256)
model_2 = RetinaModel('Retinal_OCT_RESNET50-05092022-20-57.tflite', 'RESNET50', 256)
model_3 = RetinaModel('Retinal_OCT_VGG19-05172022-21-38.tflite', 'VGG19', 256)
def get_category(img):
    """Write a Function to Predict the Class Name

    Args:
        img [jpg]: image file

    Returns:
        [str]: Prediction
    """
    logger.info("calling model.predict method")
    pred1 = model_1.predict(img)
    pred2 = model_2.predict(img)
    pred3 = model_3.predict(img)
    ret_val = '{}: {}; {}: {}; {}: {}'.format(model_1.get_model_description(), pred1,model_2.get_model_description(),pred2,model_3.get_model_description(),pred3)
    return ret_val


def plot_category(img):
    """Plot the input image

    Args:
        img [jpg]: image file
    """
    read_img = mpimg.imread(img)
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(ROOT_DIR + '/static/images/output.png')
    plt.imsave(path, read_img)