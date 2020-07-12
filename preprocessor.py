from keras.utils import to_categorical
from keras import backend as K

from config import Config 


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

def preprocess_image(image: list) -> list:    
            
    # reshape and normalize between 0 and 1 (black-white color coding)
    image = image.reshape(image.shape[0], 1, Config.img_rows, Config.img_cols)
    image = image.astype(Config.X_astype)
    image /= 255
    return image