from joblib import dump
from joblib import load


def save_model(model_obj, file_name):
    dump(model_obj, file_name)


def load_model(file_name):
    return load(file_name)
