import logging
import mlflow
import pandas as pd
from zenml import step

from src.model_dev import LogisticRegressionModel
from sklearn.linear_model._logistic import LogisticRegression


from zenml.client import Client

experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def train_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    config: str
    ) -> LogisticRegression:
    """Trains the model on the ingested data

    Args:
        X_train (pd.DataFrame)
        X_test (pd.DataFrame)
        y_train (pd.DataFrame)
        y_test (pd.DataFrame)

    Returns:
        RegressorMixin: the trained model
    """

    model = None
    if config == "LogisticRegressionModel":
        mlflow.sklearn.autolog()
        model = LogisticRegressionModel()
        train_model = model.train(X_train, y_train)
        return train_model
    else:
        raise ValueError(f"Model {config} not supported")
        
    