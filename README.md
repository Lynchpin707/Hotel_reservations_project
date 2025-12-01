# End-to-end MLOps pipeline - Predicting Hotel reservation cancellation

## Project overview

While building this project, my main focus was not the model itself, but all the engineering behind it. I wanted to build a reproducible and orchestrated ML pipeline.

## Tech stack

- **Python** for core development and in association with :
    - **ZenML** as the orchestration backbone of the project.
    - **Pandas** and **NumPy** for data manipulation and numerical processing.
    - **Scikit-learn** for model development and evaluation.
- **Git** for rigorous version control.

## Project objectives   

My objectives while building this project were simple and clear :

- To build a functionnal end-to-end ML pipeline that successfuly predicts hotel reservation cancellation.
- To insure modularity and maintability with the strategy design pattern.
- To use ZenML for orchestration.
- To apply structured logging, error handling and rigorous version control.
- To apply Clean Code practices.

and finally :
- To well document the project, which is what this repository is for.

## Pipeline architecture

Our workflow is split into simple, independent and swappable components (or steps in this case):

**1) Data Ingestion :** Loading raw hotel reservation data from a csv file.

**2) Data Cleaning :** Feature selection and deviding the data to training and test sets.(Implemented as a strategy to allow multiple cleaning methods)

**3) Model Development :** Training the ML model using the configurable model strategies (ex: Logistic regression).

**4) Model Evaluation :** Evaluation strategy that allows easy extension and use of multiple metrics (Accuracy, precision, recall, F1-score.)

<img src="docs/ZenML pipeline.png" alt="The pipeline architecture on ZenML" width="800"/>

The pipeline is orchestrated by ZenML, ensuring orderly execution, caching, reproducibility, and clean deployment paths. Also, every step handles its own validation and logging.

## How to run :



```bash
    git clone https://github.com/Lynchpin707/Hotel_reservations_project#
```

Install the dependencies:

```bash
    pip install -r requirements.txt 
```
Run the ZenML pipeline:

```bash
zenml init
python run_pipeline.py
```
ZenML will then provide you with a dashboard you can visualise on your local machine.

## Ressources:
- ["How to write software documentation"](https://www.writethedocs.org/guide/writing/beginners-guide-to-docs/) : An article that helped me figure out how to write good documentation for my engineering projects.

- [freeCodeCamp MLOps Course](https://www.youtube.com/watch?v=-dJPoLm_gtE) : A very nice course to learn and practice applying DevOps principles to machine learning.

- [Kaggle Hotel Reservations Dataset](https://www.kaggle.com/datasets/ahsan81/hotel-reservations-classification-dataset) : The main data source.