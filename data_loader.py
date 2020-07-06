from keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()

def load_data(datapart: str) -> list:
        
    if datapart == 'train':
        output = X_train[:50000], y_train[:50000]
        
    elif datapart == 'train_add':
        output = X_train[50000:], y_train[50000:]
        
    elif datapart == 'test':
        output = X_test, y_test
    
    return  output