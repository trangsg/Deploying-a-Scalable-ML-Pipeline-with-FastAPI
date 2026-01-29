import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from ml.model import train_model, inference, compute_model_metrics
from ml.data import process_data

# TODO: add necessary import
# Sample data for testing
def sample_data():
    data = pd.DataFrame({
        "age": [25, 38, 28, 44],
        "workclass": ["Private", "Self-emp-not-inc", "Private", "Private"],
        "education": ["Bachelors", "HS-grad", "HS-grad", "Some-college"],
        "marital-status": ["Never-married", "Married-civ-spouse", "Divorced", "Married-civ-spouse"],
        "occupation": ["Tech-support", "Craft-repair", "Other-service", "Exec-managerial"],
        "relationship": ["Not-in-family", "Husband", "Unmarried", "Husband"],
        "race": ["White", "Black", "White", "White"],
        "sex": ["Male", "Male", "Female", "Male"],
        "hours-per-week": [40, 50, 30, 60],
        "native-country": ["United-States", "United-States", "United-States", "United-States"],
        "salary": [0, 1, 0, 1]
    })
    return data
# TODO: implement the first test. Change the function name and input as needed
def test_train_model_returns_model():
    """
    # Test 1: Train model returns a classifier
    """
    # Your code here
    data = sample_data()
    X = data[["age", "hours-per-week"]]
    y = data["salary"]
    model = train_model(X, y)
    assert isinstance(model, RandomForestClassifier)


# TODO: implement the second test. Change the function name and input as needed
def test_inference_returns_array():
    """
    #  Test 2: Inference returns predictions of correct shape
    """
    # Your code here
    data = sample_data()
    X = data[["age", "hours-per-week"]]
    y = data["salary"]
    model = train_model(X, y)
    preds = inference(model, X)
    assert isinstance(preds, np.ndarray)
    assert preds.shape == y.shape


# TODO: implement the third test. Change the function name and input as needed
def test_compute_model_metrics_types():
    """
    # Test 3: compute_model_metrics returns floats
    """
    # Your code here
    y = np.array([0, 1, 0, 1])
    preds = np.array([0, 1, 0, 0])
    precision, recall, fbeta = compute_model_metrics(y, preds)
    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(fbeta, float)