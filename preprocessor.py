from keras.utils import to_categorical
from keras import backend as K

from cv2 import cv2
import numpy as np
import base64

from config import Config

#Initialize the useless part of the base64 encoded image.
init_Base64 = 21


def preprocess_x(X_dataset: list) -> list:
    
    # reshape and normalize between 0 and 1 (black-white color coding)
    X_dataset = X_dataset.reshape(X_dataset.shape[0], 1, 
                                  Config.img_rows, Config.img_cols)
    X_dataset = X_dataset.astype(Config.X_astype)
    X_dataset /= 255
    return X_dataset
    
def preprocess_y(y_dataset: list, num_classes: int) -> list:
    
    # class vectors to binary class matrices (one-hot encoding)
    y_dataset = to_categorical(y_dataset, Config.num_classes)    
    return y_dataset

def preprocess_image(image) -> list:    
    
    image = np.asarray(bytearray(image), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)
    
    #Resizing and reshaping to keep the ratio
    resized = cv2.resize(image, (28,28), interpolation = cv2.INTER_AREA)
    vect = np.asarray(resized, dtype="uint8")
    vect = vect.reshape(1, 1, 28, 28).astype('float32')/255
    
    return vect