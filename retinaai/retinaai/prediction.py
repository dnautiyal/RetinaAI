import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from .model import RetinaModel
import logging
# logging.basicConfig(filename='/opt/python/log/retinaai.log', level=logging.DEBUG)
logger = logging.getLogger(__name__)
deepak_model = RetinaModel('Retinal_OCT_INCEPTIONV3-04062022-00-33.tflite', 'INCEPTIONV3', 256)
def get_category(img):
    """Write a Function to Predict the Class Name

    Args:
        img [jpg]: image file

    Returns:
        [str]: Prediction
    """
    logger.info("calling model.predict method")
    pred = deepak_model.predict(img)
    return pred


def plot_category(img):
    """Plot the input image

    Args:
        img [jpg]: image file
    """
    read_img = mpimg.imread(img)
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(ROOT_DIR + '/static/images/output.png')
    plt.imsave(path, read_img)