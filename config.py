import os

class Config(object):
    # SECRET_KEY = os.environ.get('SECRET_KEY') or 'you-will-never-guess'
    img_rows = 28
    img_cols = 28
    input_shape = (1, img_rows, img_cols)
    num_classes = 10
    batch_size = 128
    epochs = 12
    X_astype = 'float32'