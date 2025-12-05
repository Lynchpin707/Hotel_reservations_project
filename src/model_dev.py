import logging
from abc import ABC, abstractmethod
from sklearn.linear_model import LogisticRegression

class Model(ABC):
    
    @abstractmethod
    def train(self, X_train, y_train):
        """Trains the model

        Args:
            X_train (_type_): training data
            y_train (_type_): training labels
        """
        pass
    
class LogisticRegressionModel(Model):
    """Logistic regression model
    """
    def train(self, X_train, y_train, **kwargs):
        """Trains the model

        Args:
            X_train (_type_): training data
            y_train (_type_): training labels
        """
        try:
            
            logisticreg = LogisticRegression(max_iter=1000, **kwargs) 
            logisticreg.fit(X_train, y_train)
            logging.info("Model training completed")
            return logisticreg
        except Exception as e:
            logging.error("Error occured while training the model: {}".format(e))
            raise e