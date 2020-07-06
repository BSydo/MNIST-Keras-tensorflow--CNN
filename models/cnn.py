import pickle

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

import preprocessor
import config
                   
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
input_shape = (config.img_rows, config.img_cols, 1)
X_train = preprocessor.preprocess_x(X_train)
X_test = preprocessor.preprocess_x(X_test)
# X_train_add = preprocessor.preprocess_x(X_train_add)

# Y data
y_train = preprocessor.preprocess_y(y_train, config.num_classes)
y_test = preprocessor.preprocess_y(y_test, config.num_classes)
# y_train_add = preprocessor.preprocess_y(y_train_add, config.num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2))) # keeping the brightest pixels
model.add(Dropout(0.25)) # fight overfitting
model.add(Flatten())
model.add(Dense(128, activation='relu')) # classifying the flatten images
model.add(Dropout(0.25))
model.add(Dense(config.num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy']
             )

model.fit(X_train, y_train,
         batch_size=config.batch_size,
         epochs=config.epochs,
         verbose=1,
         validation_data=(X_test, y_test))
score = model.evaluate(X_test, y_test, verbose=0)
print('Test Loss:', score[0])
print('Test accuracy:', score[1])

# model.fit(X_train_add, y_train_add,
#          batch_size=config.batch_size,
#          epochs=config.epochs,
#          verbose=1,
#          validation_data=(X_test, y_test))
# score = model.evaluate(X_test, y_test, verbose=0)
# print('Test ADD Loss:', score[0])
# print('Test ADD accuracy:', score[1])