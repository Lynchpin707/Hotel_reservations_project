import logging
import pandas as pd
import mlflow
from zenml import step
from zenml.client import Client



from sklearn.linear_model._logistic import LogisticRegression
from src.evaluation import AccuracyScore, F1Score, CrossEntropyLoss 
from typing import Tuple
from typing_extensions import Annotated

experiment_tracker = Client().active_stack.experiment_tracker


@step(experiment_tracker=experiment_tracker.name)
def evaluate_model(
    model: LogisticRegression,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame
    ) -> Tuple[
        Annotated[float, "accuracy"],
        Annotated[float, "f1 score"]
        ] :
    """Evaluates the trained model

    Args:
        model (RegressorMixin): the trained model
        X_test (pd.DataFrame): the test data
        y_test (pd.DataFrame): labels of the test data

    Raises:
        e: error while calculating the metrics

    Returns:
        tuple: accuracy and f1 score
    """
    try:
        
        prediction = model.predict(X_test)
        
        accuracy_class = AccuracyScore()
        accuracy = accuracy_class.calculate_scores(y_test, prediction)
        mlflow.log_metric("Accuracy score", accuracy)
        f1_class = F1Score()
        f1_score = f1_class.calculate_scores(y_test, prediction)
        mlflow.log_metric("F1 score", f1_score)
        return accuracy, f1_score
    except Exception as e:
        logging.error("Error occured during model evaluation: {}".format(e))
        raise e
    