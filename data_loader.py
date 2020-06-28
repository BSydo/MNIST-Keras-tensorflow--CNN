from keras.datasets import mnist
import pickle


(X_train, y_train), (X_test, y_test) = mnist.load_data()

files = ["datasets/test_data/x_test_data.txt",
         "datasets/test_data/y_test_data.txt",
         "datasets/train_data/x_train_data.txt",
         "datasets/train_data/y_train_data.txt",
         "datasets/train_data_add/x_train_data_add.txt",
         "datasets/train_data_add/y_train_data_add.txt"
         ]

data = [X_train[:50000],
        y_train[:50000],
        X_train[50000:],
        y_train[50000:],
        X_test,
        y_test
        ]

for ind in range(len(files)):    
    with open(files[ind], 'wb') as fp:
        pickle.dump(data[ind], fp)