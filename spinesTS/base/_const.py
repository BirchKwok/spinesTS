import numpy as np
import pandas as pd


class DataTS:
    """provide common code for spinesTS.data module. """
    def __init__(self, dataset):
        assert isinstance(dataset, pd.DataFrame) or isinstance(dataset, np.ndarray)
        if isinstance(dataset, pd.DataFrame) is False:
            self._dataset = pd.DataFrame(dataset)
        self._dataset = dataset

    @property
    def data(self):
        return self._dataset

    @property
    def head_data(self):
        return self._dataset.head()

    @property
    def shape(self):
        return self._dataset.shape

    def __str__(self):
        return f"spinesTS.DataTS(shape={np.shape(self._dataset)}, head_data=\n{self._dataset.head()})"

    def __repr__(self):
        return self.__str__()

