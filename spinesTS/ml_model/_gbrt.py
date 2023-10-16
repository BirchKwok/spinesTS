import numpy as np
import pandas as pd

from spinesTS.base import ForecastingMixin
from spinesTS.ml_model import MultiOutputRegressor
from spinesTS.utils import check_is_fitted
from spinesTS.pipeline import Pipeline
from spinesTS.preprocessing import split_series, lag_splits, moving_average
from spinesTS.features_generator import DataExtendFeatures


class GBRTPreprocessing:
    """WideGBRT features engineering class"""
    def __init__(self, in_features, out_features, target_col,
                 train_size=0.8, date_col=None, differential_n=0, moving_avg_n=0,
                 extend_daily_target_features=True
                 ):
        self.cf = None
        self.input_features = in_features
        self.output_features = out_features
        self.target_col = target_col
        self.train_size = train_size
        self.date_col = date_col
        self.extend_daily_target_features = extend_daily_target_features
        assert isinstance(differential_n, int) and differential_n >= 0
        self.differential_n = differential_n
        assert isinstance(moving_avg_n, int) and moving_avg_n >= 0
        self.moving_avg_n = moving_avg_n
        self.x_shape = None

        self.__spinesTS_is_fitted__ = False

    def process_date_col(self, x):
        """Processing date column"""
        return DataExtendFeatures(date_col=self.date_col, drop_date_col=True).fit_transform(x)

    @staticmethod
    def process_target_col(x):
        assert x.ndim == 2

        mean_res = x.mean(axis=1).reshape((-1, 1))
        median_res = np.percentile(x, q=50, axis=1).reshape((-1, 1))
        min_res = x.min(axis=1).reshape((-1, 1))
        max_res = x.max(axis=1).reshape((-1, 1))
        p25 = np.percentile(x, q=25, axis=1).reshape((-1, 1))
        p75 = np.percentile(x, q=75, axis=1).reshape((-1, 1))
        std = np.std(x, axis=1).reshape((-1, 1))

        return np.concatenate((mean_res, median_res, max_res, min_res, p25, p75, std), axis=1)

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
        self.x_shape = x.shape[1]
        self.__spinesTS_is_fitted__ = True
        return self

    def transform(self, x, mode='train'):
        """Transform data to fit WideGBRT model.

        result's columns sequence:
        lag_1, lag_2, lag_3, ..., lag_n, x_col_1, x_col_2, ..., x_col_n, date_fea_1, date_fea_2, ..., date_fea_n

        Parameters
        ----------
        x: spines.data.DataTS or pandas.core.DataFrame or numpy.ndarray, the data that needs to be transformed
        mode: ('train', 'predict'), the way to transform data, default: 'train'

        Returns
        -------
        numpy.ndarray, x_train, x_test, y_train, y_test, when mode = 'train', else, x, y

        """
        assert mode in ('train', 'predict')
        check_is_fitted(self)

        self.check_x_types(x)

        if x.shape[1] != self.x_shape:
            raise ValueError("data shape does not match the shape of the data at the time of fitting.")

        _tar = x[self.target_col].values if isinstance(x, pd.DataFrame) else x[:, self.target_col]

        if isinstance(x, pd.DataFrame):
            if self.date_col is not None:
                x = self.process_date_col(x)
            # timestamp features
            _non_lag_fea = x.loc[:, [i for i in x.columns if i != self.target_col]].values
        else:
            if self.date_col is not None:
                x = self.process_date_col(pd.DataFrame(x, columns=range(x.shape[1]))).values

            _non_lag_fea = x[:, [i for i in range(x.shape[1]) if i != self.target_col]]

        if mode == 'train':
            if self.train_size is None:
                x, y = split_series(_tar, _tar, self.input_features, self.output_features, train_size=self.train_size)

                if self.extend_daily_target_features:
                    tar_fea_x = self.process_target_col(x)

                if self.moving_avg_n > 0:
                    x = moving_average(x, window_size=self.moving_avg_n)

                if self.differential_n > 0:
                    x = np.diff(x, axis=1, n=self.differential_n)

                x_non_lag, _ = split_series(_non_lag_fea, _tar, self.input_features,
                                            self.output_features, train_size=self.train_size)

                if x_non_lag.shape[1] > 0:
                    if self.extend_daily_target_features:
                        return np.concatenate((x, tar_fea_x, self._process_x_non_lag_dim(x_non_lag)), axis=1), y
                    return np.concatenate((x, self._process_x_non_lag_dim(x_non_lag)), axis=1), y
                else:
                    if self.extend_daily_target_features:
                        return np.concatenate((x, tar_fea_x), axis=1), y
                    return x, y
            else:
                x_train, x_test, y_train, y_test = split_series(_tar, _tar, self.input_features,
                                                                self.output_features, train_size=self.train_size)

                if self.extend_daily_target_features:
                    tar_fea_x_train = self.process_target_col(x_train)
                    tar_fea_x_test = self.process_target_col(x_test)

                if self.moving_avg_n > 0:
                    x_train = moving_average(x_train, window_size=self.moving_avg_n)
                    x_test = moving_average(x_test, window_size=self.moving_avg_n)

                if self.differential_n > 0:
                    x_train = np.diff(x_train, axis=1, n=self.differential_n)
                    x_test = np.diff(x_test, axis=1, n=self.differential_n)

                x_non_lag_train, x_non_lag_test, _, _ = split_series(_non_lag_fea, _tar, self.input_features,
                                                                     self.output_features, train_size=self.train_size)
                # columns sequence:
                # lag_1, lag_2, lag_3, ..., lag_n, x_col_1, x_col_2, ..., x_col_n,
                # date_fea_1, date_fea_2, ..., date_fea_n
                if len(x_non_lag_train) > 0 and len(x_non_lag_test) > 0:
                    if self.extend_daily_target_features:
                        return np.concatenate((x_train, tar_fea_x_train, self._process_x_non_lag_dim(x_non_lag_train)), axis=1), \
                            np.concatenate((x_test, tar_fea_x_test, self._process_x_non_lag_dim(x_non_lag_test)), axis=1), y_train, y_test
                    return np.concatenate((x_train, self._process_x_non_lag_dim(x_non_lag_train)), axis=1), \
                        np.concatenate((x_test, self._process_x_non_lag_dim(x_non_lag_test)), axis=1), y_train, y_test
                else:
                    if self.extend_daily_target_features:
                        np.concatenate((x_train, tar_fea_x_train), axis=1), \
                            np.concatenate((x_test, tar_fea_x_test), axis=1), y_train, y_test
                    return x_train, x_test, y_train, y_test
        else:
            split_tar = lag_splits(
                _tar, window_size=self.input_features, skip_steps=1, pred_steps=1
            )[:-self.output_features]

            if self.extend_daily_target_features:
                tar_fea_x = self.process_target_col(split_tar)

            if self.moving_avg_n > 0:
                split_tar = moving_average(split_tar, window_size=self.moving_avg_n)

            if self.differential_n > 0:
                split_tar = np.diff(split_tar, axis=1, n=self.differential_n)

            split_non_lag_fea = lag_splits(
                _non_lag_fea, window_size=self.input_features, skip_steps=1, pred_steps=1
            )[:-self.output_features]

            if len(split_non_lag_fea) > 0:
                if self.extend_daily_target_features:
                    return np.concatenate((split_tar, tar_fea_x, self._process_x_non_lag_dim(split_non_lag_fea)), axis=1)
                return np.concatenate((split_tar, self._process_x_non_lag_dim(split_non_lag_fea)), axis=1)
            else:
                if self.extend_daily_target_features:
                    return np.concatenate((split_tar, tar_fea_x), axis=1)
                return split_tar

    @staticmethod
    def _process_x_non_lag_dim(x):
        if x[:, -1, :].squeeze().ndim == 1 and x[:, -1, :].ndim == 2:
            return x[:, -1, :]
        elif x[:, -1, :].squeeze().ndim == 1:
            return x[:, -1, :].squeeze(1)
        return x[:, -1, :].squeeze()

class WideGBRT(ForecastingMixin):
    def __init__(self, model, scaler=None, is_pipeline=False):
        """
        
        Parameters
        -----------
        model: estimator, note that estimator should implement the fit and predict method.
        scaler: Numerical scale normalizer, note that scaler should implement the fit and predict method.
            Default to None.
        is_pipeline: whether model is a sklearn-type pipeline,
            if not, class WideGBRT will automatically assemble a pipeline to predict multiple target values.
            Default to False.

        Returns
        -------
        None

        """
        if not is_pipeline:
            if scaler:
                multi_reg = Pipeline([
                    ('sc', scaler),
                    ('multi_reg', MultiOutputRegressor(model))
                ])
    
                self.model = multi_reg
            else:
                self.model = MultiOutputRegressor(model)
        else:
            self.model = model

        self.__spinesTS_is_fitted__ = False

    def fit(self, X, y, **model_fit_kwargs):
        self.model.fit(X, y, **model_fit_kwargs)

        self.__spinesTS_is_fitted__ = True
        return self

    def predict(self, X, **model_predict_kwargs):
        check_is_fitted(self)
        return self.model.predict(X, **model_predict_kwargs)
