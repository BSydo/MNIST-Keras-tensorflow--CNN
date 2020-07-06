from keras.utils import to_categorical
from keras import backend as K

import config


def preprocess_x(X_dataset: list) -> list:
    
    # reshape and normalize between 0 and 1 (black-white color coding)
    X_dataset = X_dataset.reshape(X_dataset.shape[0], 1, 
                                  config.img_rows, config.img_cols)
    X_dataset = X_dataset.astype(config.X_astype)
    X_dataset /= 255
    return X_dataset
    
def preprocess_y(y_dataset: list, num_classes: int) -> list:
    
    # class vectors to binary class matrices (one-hot encoding)
    y_dataset = to_categorical(y_dataset, config.num_classes)    
    return y_dataset
