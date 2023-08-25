import numpy as np, copy
import pandas as pd
from scipy import stats
from joblib import Parallel, delayed
from sklearn.linear_model import LinearRegression
from spinesUtils.feature_tools import vars_threshold, variation_threshold

from spinesTS.utils import check_is_fitted
from spinesTS.preprocessing import lag_splits


class ContinuousFeatureGenerator:
    """Extract features for two dims series.

    Parameters
    ----------
    window_size : int or float. The length of the interval around the outlier,
            if float-type, represents the ratio of the range of each move to the length of each data sample;
            if integer-type, represents the step size of each move.
    drop_init_features : bool, whether to drop the initial features

    Returns
    -------
    None
    """

    def __init__(
            self,
            drop_init_features=False,
            columns_prefix='continuous_feature_'
    ):
        """
        Extract features for continuous sequences of inputs.
        """

        self._drop_init_features = drop_init_features
        self.__spinesTS_is_fitted__ = False

        self.columns_prefix = columns_prefix
        self.columns = []

    def get_usual_statistical(self, x):
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

        return np.concatenate((_mean, _kurt, _skew, _min, _max, _median, _var), axis=-1), [
            self.columns_prefix + 'mean', self.columns_prefix + 'kurt', self.columns_prefix + 'skew',
            self.columns_prefix + 'min', self.columns_prefix + 'max', self.columns_prefix + 'median',
            self.columns_prefix + 'var'
        ]

    def get_linearity(self, x):
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

        return np.concatenate((np.array(_coeff), np.array(_residual)), axis=-1), [
            self.columns_prefix + 'coeff', self.columns_prefix + 'residual'
        ]

    def get_entropy(self, x):
        """Get the cross entropy features.

        Parameters
        ----------
        x : array-like

        Returns
        -------
        numpy.ndarray
        """
        _ = Parallel(n_jobs=-1)(delayed(stats.entropy)(_) for _ in x)
        _ = np.array(_).reshape((-1, 1))
        res = np.where((np.isinf(_) | np.isnan(_)), 0., _)

        return res, [
            self.columns_prefix + 'entropy'
        ]

    def fit(self, x):
        """Fit the inputting matrix.

        Parameters
        ----------
        x : array-like

        Returns
        -------
        self
        """

        assert isinstance(x, np.ndarray) and x.ndim == 2
        self.__spinesTS_is_fitted__ = True
        return self

    def fit_transform(self, x):
        """
        Fit and extract the features of the inputting matrix.

        Parameters
        ----------
        x : array-like
        inplace : Whether to transform x in place or to return a copy.

        Returns
        -------
        numpy.ndarray
        """
        check_is_fitted(self)
        self.fit(x)

        return self.transform(x)

    def transform(self, X):
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

        gus = self.get_usual_statistical(x)
        ge = self.get_entropy(x)
        gl = self.get_linearity(x)
        res = [gus, ge, gl]
        for c in res:
            self.columns.extend(c[1])

        if self._drop_init_features:
            return np.concatenate([i[0] for i in res], axis=-1)
        else:
            return np.concatenate([x, *[i[0] for i in res]], axis=-1)


def date_features(df, date_col, drop_date_col=True, format=None):
    """Date features generation"""
    assert isinstance(df, pd.DataFrame)
    x = df[date_col].copy().to_frame()

    ds_col = pd.to_datetime(x[date_col], format=format)
    x['hour'] = ds_col.dt.hour
    x['minute'] = ds_col.dt.minute
    x['weekday_1'] = ds_col.dt.weekday

    x['week_1'] = ds_col.dt.week
    x['month_1'] = ds_col.dt.month
    x['weekofyear'] = ds_col.dt.weekofyear
    x['quarter_1'] = ds_col.dt.quarter
    x['day_of_week'] = ds_col.dt.dayofweek + 1
    x['day_of_month'] = ds_col.dt.day

    x['day_of_year'] = ds_col.dt.dayofyear

    def dayofquarter(s):
        if 1 <= s.month <= 3:
            return s.dayofyear
        elif 4 <= s.month <= 6:
            return s.dayofyear - (pd.to_datetime(str(s.year) + '-03-31') - pd.to_datetime(str(s.year) + '-01-01')).days - 1
        elif 7 <= s.month <= 9:
            return s.dayofyear - (pd.to_datetime(str(s.year) + '-06-30') - pd.to_datetime(str(s.year) + '-01-01')).days - 1
        else:
            return s.dayofyear - (pd.to_datetime(str(s.year) + '-9-30') - pd.to_datetime(str(s.year) + '-01-01')).days - 1

    x['dayofquarter'] = ds_col.apply(lambda s: dayofquarter(s))

    x['day_to_mid_quarter'] = x['dayofquarter'] - 15
    x['day_to_start_quarter'] = x['dayofquarter'] - 1
    x['day_to_end_quarter'] = x['dayofquarter'] - 90

    x['daytomonday'] = abs(ds_col.dt.weekday - 1)  # 仅考虑当周周一
    x['daytofriday'] = abs(ds_col.dt.weekday - 5)  # 仅考虑当周周五
    x['daytomiddleweek'] = abs(ds_col.dt.weekday - 3)  # 仅考虑当周周三

    x['is_weekend'] = ds_col.dt.weekday // 4
    x['is_startofwork'] = ds_col.apply(lambda s: 0 if s.weekday() not in [1, 2] else 1)
    x['is_endofwork'] = ds_col.apply(lambda s: 0 if s.weekday() not in [4, 5] else 1)
    x['is_startofmonth'] = ds_col.dt.is_month_start.astype(np.int8)
    x['is_middleofmonth'] = ds_col.apply(lambda s: 0 if s.day not in [14, 15, 16] else 1)
    x['is_endofmonth'] = ds_col.apply(
        lambda s: 0 if s.day not in [27, 28, 29, 30, 31] else 1)

    x['week_of_month'] = ds_col.apply(lambda d: (d.day - 1) // 7 + 1)
    x['week_of_year'] = ds_col.dt.weekofyear
    x['year_diff'] = ds_col.dt.year.max() - ds_col.dt.year
    x['is_quarter_start'] = ds_col.dt.is_quarter_start.astype(np.int8)

    x['is_year_start'] = ds_col.dt.is_year_start.astype(np.int8)
    x['is_year_end'] = ds_col.dt.is_year_end.astype(np.int8)
    # 0: Winter - 1: Spring - 2: Summer - 3: Fall
    x["season"] = np.where(ds_col.dt.month.isin([12, 1, 2]), 0, 1)
    x["season"] = np.where(ds_col.dt.month.isin([6, 7, 8]), 2, x["season"])
    x["season"] = pd.Series(np.where(ds_col.dt.month.isin([9, 10, 11]), 3, x["season"])).astype("int8")

    to_remove_col_names = list(set(vars_threshold(x) + variation_threshold(x)))
    to_remove_col_names.append(date_col)
    if len(to_remove_col_names) > 0:
        x.drop(columns=to_remove_col_names, inplace=True)

    df = pd.concat((df, x), axis=1)
    return df if not drop_date_col else df.drop(columns=date_col)


def groupby_target_features(df, target_col):
    """target features"""
    assert isinstance(df, pd.DataFrame)
    x = df[target_col].copy().to_frame()

    days = {
        'year': 365,
        'quarter': 90,
        'month': 30,
        'half_month': 15,
        'week': 7
    }
    for col in ['year', 'quarter', 'month', 'half_month', 'week']:
        d_length = days[col]
        mean_res = []
        skew_res = []
        kurt_res = []
        median_res = []
        variation_res = []

        for i in range(x.shape[0]):
            if i - d_length < 0:
                start = 0
            else:
                start = i - d_length

            if i == 0:
                mean_res.append(None)
                skew_res.append(None)
                kurt_res.append(None)
                median_res.append(None)
                variation_res.append(None)
            elif i == 1:
                mean_res.append(x[target_col].iloc[0])
                skew_res.append(None)
                kurt_res.append(None)
                median_res.append(x[target_col].iloc[0])
                variation_res.append(None)
            else:
                ts = x[target_col].iloc[start: i]
                mean_res.append(ts.mean())
                skew_res.append(ts.skew())
                kurt_res.append(ts.kurt())
                median_res.append(ts.median())
                variation_res.append(ts.std() / ts.mean())

        x[target_col+'_'+col+'_mean'] = mean_res
        x[target_col+'_'+col + '_skew'] = skew_res
        x[target_col+'_'+col + '_kurt'] = kurt_res
        x[target_col+'_'+col + '_median'] = median_res
        x[target_col+'_'+col + '_variation'] = variation_res

    to_remove_col_names = list(set(vars_threshold(x) + variation_threshold(x)))
    to_remove_col_names.append(target_col)
    if len(to_remove_col_names) > 0:
        x.drop(columns=to_remove_col_names, inplace=True)

    df = pd.concat((df, x), axis=1)

    return df


class TableFeatureGenerator:
    """Table features """

    def __init__(self,
                 target_col,
                 window_size,
                 drop_init_features=False,
                 date_col=None
                 ):
        self.target_col = target_col
        self.date_col = date_col
        self._window_size = window_size

        self.__spinesTS_is_fitted__ = False
        self.continuous_fe = ContinuousFeatureGenerator(
            drop_init_features=drop_init_features,
        ).fit(np.ones((1, 1)))

    @staticmethod
    def _split_target(x, window_size, skip_steps=1):
        return lag_splits(x, window_size=window_size, skip_steps=skip_steps, pred_steps=1)

    def fit(self, X):
        assert isinstance(X, pd.DataFrame), \
            "parameter `X` only accept the pandas DataFrame-type data. "

        self.__spinesTS_is_fitted__ = True
        return self

    def transform(self, X, fillna=False):
        check_is_fitted(self)
        x = X.copy()
        x = groupby_target_features(x, self.target_col)
        if self.date_col is not None:
            x = date_features(x, self.date_col)

        lag_features = self._split_target(x[self.target_col], window_size=self._window_size, skip_steps=1)

        _ = self.continuous_fe.transform(lag_features)
        lag_features = pd.DataFrame(
            _, columns=[f'{self.target_col}_{i}' for i in range(self._window_size)] +
            self.continuous_fe.columns
        )

        # fill the first rows
        fillna = fillna if fillna is not False else np.nan
        _ = pd.DataFrame([[fillna for i in range(lag_features.shape[1])] for j in range(self._window_size)],
                         columns=lag_features.columns)

        lag_features = pd.concat((_, lag_features), ignore_index=True, axis=0)

        return pd.concat((x, lag_features.iloc[:-1, :]), axis=1)

    def fit_transform(self, X, fillna=False):
        self.fit(X)
        return self.transform(X, fillna=fillna)
