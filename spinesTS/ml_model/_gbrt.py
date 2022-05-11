import numpy as np
import pandas as pd
from lightgbm import LGBMModel
from sklearn.preprocessing import StandardScaler
from spinesTS.base import MLModelMixin
from spinesTS.ml_model import MultiOutputRegressor
from spinesTS.utils import check_is_fitted
from spinesTS.pipeline import Pipeline
from spinesTS.preprocessing import split_series
from spinesTS.features_extractor import ContinuousFeatureExtractor, date_features


def preprocessing_gbrt_with_generate_features(X, input_features, output_features, target_col,
                                              train_size=0.8,
                                              category_cols=None, outliers_window_size=0.15, date_col=None):
    assert isinstance(X, pd.DataFrame)
    assert np.sum(np.isnan(X[target_col].values)) in [0, output_features]

    if hasattr(category_cols, '__getitem__') and not issubclass(type(category_cols), dict):
        if isinstance(category_cols[0], str):
            cate_cols = [X.columns.tolist().index(i) for i in category_cols]
        elif isinstance(category_cols[0], int):
            cate_cols = [i for i in category_cols]
        else:
            raise ValueError("category_cols only accepts an int-type or a str-type sequence.")
    elif category_cols:
        if isinstance(category_cols, int):
            cate_cols = [category_cols]
        elif isinstance(category_cols, str):
            cate_cols = [X.columns.tolist().index(category_cols)]
        else:
            raise ValueError("category_cols only accepts int or str type or a sequence which has __getitem__ attribute "
                             "and is not a dict-type.")
    if category_cols:
        _cate_fea = X.iloc[:, cate_cols].values

    # non-lag features
    _non_lag_fea = X.loc[:, ~X.columns.str.contains(target_col)]
    if date_col:
        assert date_col in _non_lag_fea.columns, f"{date_col} not in `X` columns."
        _non_lag_fea = date_features(_non_lag_fea, date_col, drop_init_features=True)
    _non_lag_fea = _non_lag_fea.values

    _tar = X[target_col].values

    x_train, x_test, y_train, y_test = split_series(_tar, _tar, input_features, output_features, train_size)
    x_train_non_lag, x_test_non_lag, _, _ = split_series(_non_lag_fea, _tar, input_features, output_features,
                                                         train_size)
    if category_cols:
        x_train_cate, x_test_cate, _, _ = split_series(_cate_fea, _cate_fea, input_features, output_features,
                                                       train_size)

    cf = ContinuousFeatureExtractor(window_size=outliers_window_size)
    x_train = cf.fit_transform(x_train)
    x_test = cf.transform(x_test)

    x_train = np.concatenate((x_train, x_train_non_lag[:, -1, :].squeeze()), axis=1)
    x_test = np.concatenate((x_test, x_test_non_lag[:, -1, :].squeeze()), axis=1)

    if category_cols:
        _shape = x_train.shape[1]
        x_train = np.concatenate((x_train, x_train_cate[:, -1, :].squeeze()), axis=1)
        x_test = np.concatenate((x_test, x_test_cate[:, -1, :].squeeze()), axis=1)
        cate_cols = [i for i in range(_shape, x_train.shape[1])]

        return cate_cols, (x_train, x_test, y_train, y_test)

    return x_train, x_test, y_train, y_test


def preprocessing_gbrt(X, input_features, output_features, target_col):
    assert isinstance(X, (pd.DataFrame, np.ndarray))
    if isinstance(X, pd.DataFrame):
        _non_lag_fea = X.loc[:, ~X.columns.str.contains(target_col)]
    else:
        assert X.ndim == 2, "Only accept two-dim numpy.ndarray."
        _non_lag_fea = X[:, [i for i in range(X.shape[1]) if i != target_col]]

    _tar = X[target_col].values if isinstance(X, pd.DataFrame) else X[:, target_col]
    x, y = split_series(_tar, _tar, input_features, output_features)
    x_non_lag, _ = split_series(_non_lag_fea, _tar, input_features, output_features)

    return np.concatenate((x, x_non_lag[:, -1, :].squeeze()), axis=1), y


class WideGBRT(MLModelMixin):
    def __init__(self, boosting_type='gbdt', num_leaves=31, max_depth=-1,
                 n_estimators=500, metric='mae', reg_alpha=0., reg_lambda=0.,
                 learning_rate=0.025, verbose=0, random_state=None, **lgb_kwargs
                 ):
        self.model = None
        self.__spinesTS_is_fitted__ = False
        self.params = {
            'boosting_type': boosting_type,
            'objective': 'regression',
            'max_depth': max_depth,
            'metric': metric,
            'reg_alpha': reg_alpha,
            'reg_lambda': reg_lambda,
            'num_leaves': num_leaves,
            'learning_rate': learning_rate,
            'verbose': verbose,
            'random_state': random_state,
            'n_estimators': n_estimators,
            **lgb_kwargs
        }

    def fit(self, X, y, scaler=StandardScaler(), sample_weight=None, init_score=None, group=None, eval_set=None,
            eval_names=None, eval_sample_weight=None, eval_class_weight=None, eval_init_score=None,
            eval_group=None, eval_metric=None, feature_name='auto',
            categorical_feature='auto', callbacks=None, init_model=None
            ):
        if callbacks is None:
            from lightgbm.callback import early_stopping
            callbacks = [early_stopping(100)]

        multi_reg = Pipeline([
            ('sc', scaler),
            ('multi_reg', MultiOutputRegressor(LGBMModel(**self.params)))
        ])
        multi_reg.fit(X, y, sample_weight=sample_weight, init_score=init_score, group=group
                      , eval_set=eval_set, eval_names=eval_names, eval_sample_weight=eval_sample_weight,
                      eval_class_weight=eval_class_weight, eval_init_score=eval_init_score,
                      eval_group=eval_group, eval_metric=eval_metric, feature_name=feature_name,
                      categorical_feature=categorical_feature, init_model=init_model, callbacks=callbacks)
        self.model = multi_reg

        self.__spinesTS_is_fitted__ = True
        return self

    def predict(self, X):
        check_is_fitted(self)
        return self.model.predict(X)
