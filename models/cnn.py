import pickle

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

import preprocessor
from config import num_classes, batch_size, epochs
# input image dimensions
from config import img_rows, img_cols

K.image_data_format() == 'channels_first'
                    
print('read pickled datasets')

# read pickled datasets
with open ("datasets/train_data/x_train_data.txt", 'rb') as fp:
    X_train = pickle.load(fp)
    fp.close()

with open ("datasets/train_data/y_train_data.txt", 'rb') as fp:
    y_train = pickle.load(fp)
    fp.close()

with open ("datasets/test_data/x_test_data.txt", 'rb') as fp:
    X_test = pickle.load(fp)
    fp.close()

with open ("datasets/test_data/y_test_data.txt", 'rb') as fp:
    y_test = pickle.load(fp)
    fp.close()

# with open ("datasets/train_data_add/x_train_data_add.txt", 'rb') as fp:
#     X_train_add = pickle.load(fp)
#     fp.close()

# with open ("datasets/train_data_add/y_train_data_add.txt", 'rb') as fp:
#     y_train_add = pickle.load(fp)
#     fp.close()

print('preprocess datasets')

# X data
X_train, input_shape = preprocessor.preprocess_x(X_train, img_cols, img_rows) 
X_test = preprocessor.preprocess_x(X_train, img_cols, img_rows)[0]
# X_train_add = preprocessor.preprocess_x(X_train_add, img_cols, img_rows)

# Y data
y_train = preprocessor.preprocess_y(y_train, num_classes)
y_test = preprocessor.preprocess_y(y_train, num_classes)
# y_train_add = preprocessor.preprocess_y(y_train_add, num_classes)


# model = Sequential()
# model.add(Conv2D(32, kernel_size=(3, 3),
#                  activation='relu',
#                  input_shape=input_shape))
# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2))) # keeping the brightest pixels
# model.add(Dropout(0.25)) # fight overfitting
# model.add(Flatten())
# model.add(Dense(128, activation='relu')) # classifying the flatten images
# model.add(Dropout(0.25))
# model.add(Dense(num_classes, activation='softmax'))

# model.compile(loss=keras.losses.categorical_crossentropy,
#               optimizer=keras.optimizers.Adadelta(),
#               metrics=['accuracy']
#              )

# model.fit(X_train, y_train,
#          batch_size=batch_size,
#          epochs=epochs,
#          verbose=1,
#          validation_data=(X_test, y_test))
# score = model.evaluate(X_test, y_test, verbose=0)
# print('Test Loss:', score[0])
# print('Test accuracy:', score[1])