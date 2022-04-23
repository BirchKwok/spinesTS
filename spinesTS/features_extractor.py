import numpy as np, copy
import pandas as pd
from scipy import stats
from joblib import Parallel, delayed
from sklearn.linear_model import LinearRegression
from spinesTS.utils import check_is_fitted


class ContinuousFeatureExtractor:
    """Extract features for two dims series.

    Returns
    -------
    None
    """

    def __init__(
            self,
            window_size=0.15,
            diff_order=1,
            drop_init_features=False,
            top_k_outlier=None
    ):
        """
        Extract features for continuous sequences of inputs.
        """
        assert isinstance(window_size, (int, float)), "window_size must be float or int"
        if isinstance(window_size, float):
            assert 0 < window_size <= 1, "window size must  be greater than 0 and less than or equal to 1" \
                                         "when the window size is a float-type number "

        self._window_size = window_size
        self._diff_order = diff_order
        self._drop_init_features = drop_init_features
        self._top_k = top_k_outlier
        self.__spinesTS_is_fitted__ = False

    @staticmethod
    def get_usual_statistical(x):
        """Get descriptive statistics.

        Parameters
        ----------
        x : array-like

        Returns
        -------
        numpy.ndarray
        """
        _mean = np.mean(x, axis=-1).reshape(-1, 1)
        _kurt = stats.kurtosis(x, axis=-1, nan_policy='omit').reshape(-1, 1)
        _skew = stats.skew(x, axis=-1, nan_policy='omit').reshape(-1, 1)
        _min = np.min(x, axis=-1).reshape(-1, 1)
        _max = np.max(x, axis=-1).reshape(-1, 1)
        _median = np.median(x, axis=-1).reshape(-1, 1)
        _var = np.var(x, axis=-1).reshape(-1, 1)

        return np.concatenate((_mean, _kurt, _skew, _min, _max, _median, _var), axis=-1)

    @staticmethod
    def get_linearity(x):
        """Get the linear fitting feature.

        Parameters
        ----------
        x : array-like

        Returns
        -------
        numpy.ndarray
        """

        def fit_each_x(_, name='coef'):
            reg = LinearRegression(n_jobs=-1)
            reg.fit(np.reshape(range(len(_)), (-1, 1)), _)
            if name == 'coef':
                return list(reg.coef_)
            else:
                pred = reg.predict(np.reshape(range(len(_)), (-1, 1)))
                residual = [np.mean(np.abs(_ - pred), axis=-1)]
                return residual

        _coeff = Parallel(n_jobs=-1)(delayed(fit_each_x)(_) for _ in x)
        _residual = Parallel(n_jobs=-1)(delayed(fit_each_x)(_, name='residual') for _ in x)

        return np.concatenate((np.array(_coeff), np.array(_residual)), axis=-1)

    @staticmethod
    def get_entropy(x):
        """Get the cross entropy features.

        Parameters
        ----------
        x : array-like

        Returns
        -------
        numpy.ndarray
        """
        _ = Parallel(n_jobs=-1)(delayed(stats.entropy)(_) for _ in x)
        _ = np.array(_).reshape(-1, 1)
        return np.where((np.isinf(_) | np.isnan(_)), 0., _)

    @staticmethod
    def get_difference(x, order=1):
        """Get the specified order difference.

        Parameters
        ----------
        x : array-like
        order : int or list or tuple, the order of difference.

        Returns
        -------
        numpy.ndarray
        """
        if isinstance(order, int):
            return np.diff(x, n=order)
        elif isinstance(order, (list, tuple)):
            return np.concatenate([np.diff(x, n=i) for i in order], axis=1)

    def get_outlier_statistical(self, x):
        """Get descriptive statistical characteristics around outliers.

        Parameters
        ----------
        x : array-like
        window_size : int or float. The length of the interval around the outlier,
            if float-type, represents the ratio of the range of each move to the length of each data sample;
            if integer-type, represents the step size of each move.

        Returns
        -------
        numpy.ndarray
        """
        if isinstance(self._window_size, float):
            window_size = int(x.shape[1] * self._window_size)
        else:
            window_size = self._window_size

        split_n = x.shape[1] // window_size
        least_n = x.shape[1] % split_n

        if least_n > 0:
            least = x[:, :least_n]
            splits = np.split(x[:, least_n:], split_n, axis=1)
            _ = [least]
            _.extend(splits)

            # only take top k samples
            _processed_list = _
        else:
            _processed_list = np.split(x, split_n, axis=1)

        if self._top_k is not None:
            _vars_to_pick = [np.var(i) for i in _processed_list]
            assert isinstance(self._top_k, int) and self._top_k > 0
            indexes = []
            _v = sorted(_vars_to_pick, reverse=True)
            for i in _v[:self._top_k]:
                for j in range(len(_vars_to_pick)):
                    if _vars_to_pick[j] == i:
                        indexes.append(j)
            _tmp_processed_list = [_processed_list[i] for i in indexes]
            _2 = Parallel(n_jobs=-1)(delayed(self.get_usual_statistical)(_) for _ in _tmp_processed_list)

        else:
            _2 = Parallel(n_jobs=-1)(delayed(self.get_usual_statistical)(_) for _ in _processed_list)

        return np.concatenate(_2, axis=-1)

    def fit(self, x):
        """Fit the inputting matrix.

        Parameters
        ----------
        x : array-like
        window_size : int or float. The length of the interval around the outlier,
            if float-type, represents the ratio of the range of each move to the length of each data sample;
            if integer-type, represents the step size of each move.
        diff_order : int or list or tuple, the order of difference.

        Returns
        -------
        self
        """

        assert isinstance(x, np.ndarray) and x.ndim == 2
        self.__spinesTS_is_fitted__ = True
        return self

    def fit_transform(self, x, inplace=False):
        """
        Fit and extract the features of the inputting matrix.

        Parameters
        ----------
        x : array-like
        window_size : int or float. The length of the interval around the outlier,
            if float-type, represents the ratio of the range of each move to the length of each data sample;
            if integer-type, represents the step size of each move.
        diff_order : int or list or tuple, the order of difference.

        Returns
        -------
        numpy.ndarray
        """
        check_is_fitted(self)
        self.fit(x)
        if inplace:
            x = self.transform(x)
        else:
            return self.transform(x)

    def transform(self, X, inplace=False):
        """Extract the features of the inputting matrix.

        Parameters
        ----------
        X : array-like

        Returns
        -------
        numpy.ndarray
        """
        check_is_fitted(self)
        x = copy.deepcopy(X)
        if inplace:
            if self._drop_init_features:
                X = np.concatenate((self.get_usual_statistical(x), self.get_entropy(x),
                                    self.get_linearity(x),
                                    self.get_outlier_statistical(x),
                                    self.get_difference(x, order=self._diff_order)), axis=-1)
            else:
                X = np.concatenate((x, self.get_usual_statistical(x), self.get_entropy(x),
                                    self.get_linearity(x),
                                    self.get_outlier_statistical(x),
                                    self.get_difference(x, order=self._diff_order)), axis=-1)
        else:
            if self._drop_init_features:
                return np.concatenate((self.get_usual_statistical(x), self.get_entropy(x),
                                       self.get_linearity(x),
                                       self.get_outlier_statistical(x),
                                       self.get_difference(x, order=self._diff_order)), axis=-1)
            else:
                return np.concatenate((x, self.get_usual_statistical(x), self.get_entropy(x),
                                       self.get_linearity(x),
                                       self.get_outlier_statistical(x),
                                       self.get_difference(x, order=self._diff_order)), axis=-1)


class TableFeatureExtractor:
    def __init__(self,
                 target_col,
                 n_lags=1,
                 window_size=0.15,
                 diff_order=1,
                 drop_init_features=False,
                 top_k_outlier=None,
                 date_col=None,
                 weighted_cross_features=True,
                 drop_multicollinearity_cols=True
                 ):
        self.target_col = target_col
        self.date_col = date_col
        self.weighted_cross_features = weighted_cross_features
        self._n_lags = n_lags
        self._window_size = window_size

        self._drop_multicollinearity_cols = drop_multicollinearity_cols
        self.__spinesTS_is_fitted__ = False
        self.continuous_fe = ContinuousFeatureExtractor(
            window_size=window_size,
            diff_order=diff_order,
            drop_init_features=drop_init_features,
            top_k_outlier=top_k_outlier
        )

    def _cross_features(self):
        pass

    def _get_linearity(self):
        pass

    def _multicollinearity_detector(self):
        pass

    def _get_n_lags(self):
        pass

    def fit(self, X):
        assert isinstance(X, (np.ndarray, pd.DataFrame)), \
            "parameter `X` only accept then pandas.DataFrame or numpy.ndarray. "
        self.__spinesTS_is_fitted__ = True
        return self

    def transform(self, X, inplace=False):
        check_is_fitted(self)
        pass

    def fit_transform(self, X, inplace=False):
        pass
