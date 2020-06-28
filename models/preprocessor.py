from keras import backend as K
from keras.utils import to_categorical

K.image_data_format() == 'channels_first'

def preprocess_x(X_dataset: list, 
                 img_cols: int, 
                 img_rows: int) -> (list,tuple):
    """[summary]

    Args:
        X_dataset (list): [description]
        img_cols (img_cols): [description]
        img_rows (img_rows): [description]

    Returns:
        list: [description]
    """

    # channels here means color coding - 1 color [1-255] in this case.
    # channels == 3 for RGB    
    if K.image_data_format() == 'channels_first': 
        X_dataset = X_dataset.reshape(X_dataset.shape[0], 1, img_rows, 
                                      img_cols
                                      )
        input_shape = (1, img_rows, img_cols)
    else:
        X_dataset = X_dataset.reshape(X_dataset.shape[0], img_rows, 
                                      img_cols, 1
                                      )
        input_shape = (img_rows, img_cols, 1)

    X_dataset = X_dataset.astype('float32')

    # normalize between 0 and 1 (black-white color coding)
    X_dataset /= 255
    
    return X_dataset, input_shape
    
    
def preprocess_y(y_dataset: list, num_classes: int) -> list:
    """[summary]

    Args:
        y_dataset (list): [description]
        num_classes ([type]): [description]

    Returns:
        list: [description]
    """
    
    # class vectors to binary class matrices (one-hot encoding)
    y_dataset = to_categorical(y_dataset, num_classes)
    
    return y_dataset
