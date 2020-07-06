import numpy as np
from keras.utils import to_categorical
import config


def preprocess_x(X_dataset: list) -> list:
    
    # reshape and normalize between 0 and 1 (black-white color coding)
    X_dataset = X_dataset[..., np.newaxis].astype(config.X_astype)/255    
    return X_dataset #, input_shape
    
    
def preprocess_y(y_dataset: list, num_classes: int) -> list:
    
    # class vectors to binary class matrices (one-hot encoding)
    y_dataset = to_categorical(y_dataset, config.num_classes)    
    return y_dataset
