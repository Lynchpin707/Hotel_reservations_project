import logging
from abc import ABC, abstractmethod
from typing import Union
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class DataStrategy(ABC):
    """ Abstract class definining strategy for handling data"""
    
    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        pass
    
class DataPreProcessStrategy(DataStrategy):
    """ Data preprocessing strategy which preprocesses the data. """
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transforms categorical data into numerical data and selects only the relevant features

        Args:
            data (pd.DataFrame): The initial df

        Raises:
            e: error occured during data preprocessing

        Returns:
            pd.DataFrame: preprocessed df
        """
        try:
            data["type_of_meal_plan"] = data["type_of_meal_plan"].map({"Meal Plan 1" : 1, "Meal Plan 2" : 2, "Meal Plan 3" : 3, "Not Selected":0 })
            
            data['room_type_reserved'] = data['room_type_reserved'].map({'Room_Type 1':1, 'Room_Type 2':2,'Room_Type 3':3,'Room_Type 4':4, 'Room_Type 5':5, 'Room_Type 6':6, 'Room_Type 7':7})
            
            data = pd.get_dummies(data, columns=['market_segment_type'], drop_first=False, dtype=int)
            
            data["booking_status"] = data["booking_status"].map({"Not_Canceled":0, "Canceled":1})
            
            data = data.drop(
                columns=['market_segment_type_Aviation', 'arrival_month', 'arrival_date', 'Booking_ID' ],
                errors='ignore' 
            )
            
            data = data.astype(float)
            
            return data
        except Exception as e:
            logging.error(f"Error occured during data preprocessing: {e}")
            raise e

        
class DataDevideStrategy(DataStrategy):
    """ Devides data into train and test sets."""
    def handle_data(self, data:pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        try:
            y = data['booking_status']
            xvalues = data.drop('booking_status', axis=1)

            X_train, X_test, y_train, y_test = train_test_split(xvalues, y, test_size=0.3, random_state=42)
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error(f"Error occured while dividing data: {e}")
            raise e
    
class DataCleaning(DataStrategy):
    """
        Data cleaning class which preprocesses the data and divides it into train and test data.
    """
    def __init__(self, data:pd.DataFrame, strategy: DataStrategy):
        self.df = data
        self.strategy = strategy
    
    def handle_data(self) -> Union[pd.DataFrame, pd.Series]:
        """
        Handles the data

        Returns:
            Union[pd.DataFrame, pd.Series]: _description_
        """
        try:
            return self.strategy.handle_data(self.df)
        except Exception as e:
            logging.error(f"Error in handling data: {e}")
            raise e
        
