import numpy as np
import pandas as pd
import json
from io import StringIO

from steps.clean_data import clean_df
from steps.evaluation import evaluate_model
from steps.ingest_data import ingest_df
from steps.model_train import train_model

from zenml import pipeline, step
from zenml.config import DockerSettings
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
from zenml.integrations.constants import MLFLOW, TENSORFLOW
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import (
    MLFlowModelDeployer,
)
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step

from .utils import get_data_for_test


docker_settings = DockerSettings(required_integration=[MLFLOW])

@step(enable_cache=False)
def dynamic_importer() -> str:
    """Downloads the latest data from a mock API."""
    data = get_data_for_test()
    return data

@step(enable_cache=False)
def prediction_service_loader(
    pipeline_name: str,
    pipeline_step_name: str,
    running: bool = True,
    model_name: str = "model"
) -> MLFlowDeploymentService:
    
    # get the MLflow deployer stack component
    mlflow_model_deployer_component = MLFlowModelDeployer.get_active_model_deployer()
    if mlflow_model_deployer_component is None:
        raise RuntimeError(
            "No active MLFlowModelDeployer found in the active stack. "
            "Make sure your active stack includes an MLflow model_deployer (e.g. `zenml integration install mlflow` and configure a stack)."
        )
    
    # fetch existing services with the right pipeline name, step name and model name
    existing_services = mlflow_model_deployer_component.find_model_server(
        pipeline_name = pipeline_name,
        pipeline_step_name = pipeline_step_name,
        model_name = model_name,
        running= running
    )
    if not existing_services:
        raise RuntimeError(
            f" No MLflow deployment service found for pipeline {pipeline_name}, step {pipeline_step_name} and model {model_name}"
        )
    return existing_services[0]
    
@step
def predictor(
    service: MLFlowDeploymentService,
    data: str # data is the JSON string from dynamic_importer
) -> np.ndarray:
    """Run an inference request against a prediction service"""

    service.start(timeout=10)  # should be a NOP if already started
    data = json.loads(data)
    data.pop("index")
    columns_for_df = data.pop("columns")
    df = pd.DataFrame(data["data"], columns=columns_for_df)
    json_list = json.loads(json.dumps(list(df.T.to_dict().values())))
    data = np.array(json_list)
    prediction = service.predict(data)
    return prediction



@step
def deployment_trigger(
    accuracy: float,
    min_accuracy: float = 0.3
):
   """
   Implements a simple model deployment trigger that verifies the model's accuracy before deciding whether to deploy it or not
   """    
   return accuracy >= min_accuracy

@pipeline(enable_cache=False, settings={"docker": docker_settings})
def continuous_deployment_pipeline(
    data_path: str,
    min_accuracy: float = 0.3,
    workers: int = 1,
    timeout: int = DEFAULT_SERVICE_START_STOP_TIMEOUT
):
    
    df = ingest_df(data_path)
    X_train, X_test, y_train, y_test = clean_df(df)
    model = train_model(X_train, X_test, y_train, y_test, "LogisticRegressionModel" )
    accuracy, f1score = evaluate_model(model, X_test, y_test)
    deployment_decision = deployment_trigger(accuracy)
    mlflow_model_deployer_step(
        model = model,
        deploy_decision = deployment_decision,
        workers = workers,
        timeout = timeout
    )

@pipeline(enable_cache=False, settings={"docker": docker_settings})
def inference_pipeline(pipeline_name: str, pipeline_step_name: str):
    # Link all the steps artifacts together
    batch_data = dynamic_importer()
    model_deployment_service = prediction_service_loader(
        pipeline_name=pipeline_name,
        pipeline_step_name=pipeline_step_name,
        running=False,
    )
    predictor(service=model_deployment_service, data=batch_data)
