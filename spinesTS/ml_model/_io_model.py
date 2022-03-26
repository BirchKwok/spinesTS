from joblib import dump
from joblib import load


def save_model(model_obj, file_name):
    """Use pickle to save python models.

    Parameters
    ----------
    model_obj: model object
    file_name: str, file-path-like

    Returns
    -------
    None
    """
    dump(model_obj, file_name)


def load_model(file_name):
    """Use pickle to load python models.

    Parameters
    ----------
    file_name: str, file-path-like

    Returns
    -------
    None
    """
    return load(file_name)
