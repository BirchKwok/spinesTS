import numpy as np

from spinesTS.base import ForecastingMixin
from spinesTS.utils import check_is_fitted


class DummyModel(ForecastingMixin):
    """
    Naive forecasting, which uses the last value of the time series to make a naive prediction

    """
    def __init__(self):
        self.in_features = None
        self.out_features = None
        self._naive_predict_results = None
        self.__spinesTS_is_fitted__ = False

    def fit(self, X, y):
        """default to ignore y"""
        self.in_features = X.shape[1]
        self.out_features = y.shape[1]
        self._naive_predict_results = X[:, -1]
        self.__spinesTS_is_fitted__ = True
        return self

    def predict(self, X=None):
        check_is_fitted(self)
        assert X.shape[1] == self.in_features

        if X is not None:
            self._naive_predict_results = X[:, -1]

        res = []

        for i in range(self.out_features):
            res.append(self._naive_predict_results.reshape((-1, 1)))

        return np.column_stack(res)
