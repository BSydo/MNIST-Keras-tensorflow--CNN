from keras.models import load_model
import os

from models import cnn
import preprocessor
import data_loader
from config import Config

# from app import app

os.environ["CUDA_VISIBLE_DEVICES"]="0"

class controller:
    
    """[summary]
    """
    
    def __init__(self):
        
        self.X_train, self.y_train = data_loader.load_data('train')
        self.X_train_add, self.y_train_add = data_loader.load_data('train_add')
        self.X_test, self.y_test = data_loader.load_data('test')
    
        self.cnn = cnn.model_init(Config.input_shape, Config.num_classes)
        self.cnn_saved = r'./models/cnn/'
        
        
    def initial_train(self) -> None:
        """[summary]
        """
        
        X_train = preprocessor.preprocess_x(self.X_train)
        y_train = preprocessor.preprocess_y(self.y_train, Config.num_classes)
        
        self.cnn.fit(X_train, y_train,
                     batch_size=Config.batch_size,
                     epochs=Config.epochs,
                     verbose=1
                     )
        
        score = self.cnn.evaluate(X_train, y_train, verbose=0)
        print('Train Loss:', score[0])
        print('Train accuracy:', score[1])
        
        print('save model')
        # Save the trained model.
        self.cnn.save(self.cnn_saved)
        del self.cnn
        
        
    def add_train(self) -> None:
        """[summary]
        """
        
        X_train_add = preprocessor.preprocess_x(self.X_train_add)
        y_train_add = preprocessor.preprocess_y(self.y_train_add, 
                                                Config.num_classes)
        # open pickled cnn model
        cnn_saved_model = load_model(self.cnn_saved)
        
        cnn_saved_model.fit(X_train_add, y_train_add,
                            batch_size=Config.batch_size,
                            epochs=Config.epochs,
                            verbose=1
                            )
        
        score = cnn_saved_model.evaluate(X_train_add,
                                         y_train_add,
                                         verbose=0
                                         )
        print('Train_add Loss:', score[0])
        print('Train_add accuracy:', score[1])
        
        print('save updated model')
        # Save the trained model as a pickle string.
        cnn_saved_model.save(self.cnn_saved)
        
        
    def test(self) -> None:
        """[summary]
        """

        X_test = preprocessor.preprocess_x(self.X_test)
        y_test = preprocessor.preprocess_y(self.y_test, Config.num_classes)

        # open pickled cnn model
        cnn_saved_model = load_model(self.cnn_saved)
        
        score = cnn_saved_model.evaluate(X_test, y_test, verbose=0)
        print('Test Loss:', score[0])
        print('Test accuracy:', score[1])
        
    def predict(self, image:list) -> int:
        """[summary]

        Args:
            image (list): [description]

        Returns:
            int: [description]
        """
        
        # image = preprocessor.preprocess_image(image)
        
        # open pickled cnn model
        cnn_saved_model = load_model(self.cnn_saved)
        
        number = cnn_saved_model.predict(image, verbose=0)
        
        return number