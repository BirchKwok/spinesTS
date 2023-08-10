import numpy as np
import pandas as pd

from spinesTS.base import MLModelMixin
from spinesTS.ml_model import MultiOutputRegressor
from spinesTS.utils import check_is_fitted
from spinesTS.pipeline import Pipeline
from spinesTS.preprocessing import split_series, lag_splits
from spinesTS.features_generator import date_features


class GBRTPreprocessing:
    # TODO: 新增gbrt特征工程处理类
    def __init__(self, input_features, output_features, target_col,
                 train_size=0.8, date_col=None):
        self.cf = None
        self.input_features = input_features
        self.output_features = output_features
        self.target_col = target_col
        self.train_size = train_size
        self.date_col = date_col

        self.__spinesTS_is_fitted__ = False

    def process_date_col(self, x):
        """Processing date column"""
        return date_features(x, date_col=self.date_col)

    def check_x_types(self, x):
        assert isinstance(x, (pd.DataFrame, np.ndarray))

        if not isinstance(x, pd.DataFrame):
            assert x.ndim == 2, "Only accept two-dim numpy.ndarray."
            if not isinstance(self.target_col, int):
                raise TypeError("when `x` is of type `numpy.ndarray`, the `target_col` parameter must be an integer.")

            if self.date_col is not None and not isinstance(self.date_col, int):
                raise TypeError("when `x` is of type `numpy.ndarray`, the `date_col` parameter must be an integer.")

    def fit(self, x):
        self.check_x_types(x)

        self.__spinesTS_is_fitted__ = True
        return self

    def transform(self, x, mode='train'):
        """Transform data to fit GBRT model.

        result's columns sequence:
        lag_1, lag_2, lag_3, ..., lag_n, x_col_1, x_col_2, ..., x_col_n, date_fea_1, date_fea_2, ..., date_fea_n
        Parameters
        ---------
        mode: ('train', 'predict'), the way to transform data, default: 'train'


        Return
        ------
        numpy.ndarray, x_train, x_test, y_train, y_test, when mode = 'train', else, x, y

        """
        assert mode in ('train', 'predict')
        check_is_fitted(self)

        self.check_x_types(x)

        if isinstance(x, pd.DataFrame):
            if self.date_col is not None:
                x = self.process_date_col(x)

            _non_lag_fea = x.loc[:, ~x.columns.str.contains(self.target_col)].values
        else:
            if self.date_col is not None:
                x = self.process_date_col(pd.DataFrame(x, columns=range(x.shape[1]))).values

            _non_lag_fea = x[:, [i for i in range(x.shape[1]) if i != self.target_col]]

        _tar = x[self.target_col].values if isinstance(x, pd.DataFrame) else x[:, self.target_col]

        if mode == 'train':
            if self.train_size is None:
                x, y = split_series(_tar, _tar, self.input_features, self.output_features, train_size=self.train_size)

                x_non_lag, _ = split_series(_non_lag_fea, _tar, self.input_features,
                                            self.output_features, train_size=self.train_size)

                return np.concatenate((x, x_non_lag[:, -1, :].squeeze()), axis=1), y
            else:
                x_train, x_test, y_train, y_test = split_series(_tar, _tar, self.input_features,
                                                                self.output_features, train_size=self.train_size)

                x_non_lag_train, x_non_lag_test, _, _ = split_series(_non_lag_fea, _tar, self.input_features,
                                                                     self.output_features, train_size=self.train_size)
                # columns sequence:
                # lag_1, lag_2, lag_3, ..., lag_n, x_col_1, x_col_2, ..., x_col_n,
                # date_fea_1, date_fea_2, ..., date_fea_n
                return np.concatenate((x_train, x_non_lag_train[:, -1, :].squeeze()), axis=1), \
                    np.concatenate((x_test, x_non_lag_test[:, -1, :].squeeze()), axis=1), y_train, y_test
        else:
            split_tar = lag_splits(
                _tar, window_size=self.input_features, skip_steps=1, pred_steps=1
            )[:-self.output_features]

            split_non_lag_fea = lag_splits(
                _non_lag_fea, window_size=self.input_features, skip_steps=1, pred_steps=1
            )[:-self.output_features]

            return np.concatenate((split_tar, split_non_lag_fea[:, -1, :].squeeze()), axis=1)


class WideGBRT(MLModelMixin):
    def __init__(self, model, scaler=None):
        if scaler:
            multi_reg = Pipeline([
                ('sc', scaler),
                ('multi_reg', MultiOutputRegressor(model))
            ])

            self.model = multi_reg
        else:
            self.model = MultiOutputRegressor(model)

        self.__spinesTS_is_fitted__ = False

    def fit(self, X, y, **model_fit_kwargs):
        self.model.fit(X, y, **model_fit_kwargs)

        self.__spinesTS_is_fitted__ = True
        return self

    def predict(self, X, **model_predict_kwargs):
        check_is_fitted(self)
        return self.model.predict(X, **model_predict_kwargs)
