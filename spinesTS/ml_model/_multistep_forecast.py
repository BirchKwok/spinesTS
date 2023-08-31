import copy
import warnings

import numpy as np
import torch
from joblib import Parallel, delayed
from sklearn.multioutput import _MultiOutputEstimator, _fit_estimator
from sklearn.utils.validation import _check_fit_params

from spinesTS.base import ForecastingMixin
from spinesTS.utils import func_has_params, check_is_fitted

warnings.filterwarnings('ignore')


class MultiStepRegressor(ForecastingMixin):
    """Use the last predict-step value as the last step true value,
        to predict current step value,
        and repeat this until it reaches y.shape[1] times.

    Parameters
    ----------
    estimator: model with implement fit and predict function

    Returns
    -------
    self : object
    """

    def __init__(self, estimator):
        self._estimator = estimator
        self._forward = 1
        self.__spinesTS_is_fitted__ = False

    def fit(self, x, y, eval_set=None, **estimator_fit_kwargs):
        assert np.ndim(y) <= 2
        if eval_set is not None:
            if len(eval_set) == 2 or len(eval_set[0]) == 2:
                if len(eval_set[0]) == 2:
                    eval_set = eval_set[0]
            else:
                raise ValueError("The eval_set must be [X, y] or [(X, y)]")

        self._forward = y.shape[1]
        _ = y
        if np.ndim(y) == 2:
            assert np.ndim(y) <= 2
            _ = y[:, 0]

        if func_has_params(self._estimator.fit, "eval_set") and eval_set is not None:

            try:
                eval_sets_ = (eval_set[0], eval_set[1][:, 0])
                self._estimator.fit(x, _, eval_set=eval_sets_, **estimator_fit_kwargs)
            except Exception:
                eval_sets_ = [(eval_set[0], eval_set[1][:, 0])]
                self._estimator.fit(x, _, eval_set=eval_sets_, **estimator_fit_kwargs)
        else:
            self._estimator.fit(x, _, **estimator_fit_kwargs)

        self.__spinesTS_is_fitted__ = True
        return self

    def predict(self, x, **kwargs):
        check_is_fitted(self)
        assert isinstance(x, np.ndarray)
        res = []
        eval_x = copy.deepcopy(x)
        for step in range(self._forward):  # forward n steps
            _ = np.squeeze(self._estimator.predict(eval_x, **kwargs))
            res.append(_)
            eval_x = np.concatenate([eval_x[:, 1:], np.transpose(_.reshape(1, -1))], axis=1)
            assert x.shape == eval_x.shape

        r = np.asarray(res)

        return np.squeeze(np.transpose(r))


class MultiOutputRegressor(ForecastingMixin, _MultiOutputEstimator):
    """Fitting one regressor per target.

    Parameters
    ----------
    estimator: model with implement fit and predict function
    n_jobs: None or int, default to None.
        The number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend`
        context. ``-1`` means using all processors.

    Returns
    -------
    self : object

    """

    def __init__(self, estimator, n_jobs=None):
        super().__init__(estimator, n_jobs=n_jobs)

    def fit(self, X, y, eval_set=None, **fit_params):
        """Fit the model to data, separately for each output variable.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.
        y : {array-like, sparse matrix} of shape (n_samples, n_outputs)
            Multi-output targets. An indicator matrix turns on multilabel
            estimation.
        eval_set : tuple of (X, y) , length must be equal to y.shape[1],
            passed to the ``estimator.fit`` method of each step
        **fit_params : dict of string -> object
            Parameters passed to the ``estimator.fit`` method of each step.

        Returns
        -------
        self : object
            Returns a fitted instance.
        """
        if eval_set is not None:
            if len(eval_set) == 2 or len(eval_set[0]) == 2:
                if len(eval_set[0]) == 2:
                    eval_set = eval_set[0]
            else:
                raise ValueError("The eval_set must be [X, y] or [(X, y)]")

        if not hasattr(self.estimator, "fit"):
            raise ValueError("The base estimator should implement a fit method")

        y = self._validate_data(X="no_validation", y=y, multi_output=True)

        if y.ndim == 1:
            raise ValueError(
                "y must have at least two dimensions for "
                "multi-output regression but has only one."
            )

        fit_params_validated = _check_fit_params(X, fit_params)

        if func_has_params(self.estimator.fit, "eval_set") and eval_set is not None:
            try:
                eval_sets_ = [(eval_set[0], eval_set[1][:, i]) for i in range(y.shape[1])]
                self.estimators_ = Parallel(n_jobs=self.n_jobs)(
                    delayed(_fit_estimator)(
                        self.estimator, X, y[:, i], eval_set=eval_sets_[i], **fit_params_validated
                    )
                    for i in range(y.shape[1])
                )
            except Exception:
                eval_sets_ = [[(eval_set[0], eval_set[1][:, i])] for i in range(y.shape[1])]
                self.estimators_ = Parallel(n_jobs=self.n_jobs)(
                    delayed(_fit_estimator)(
                        self.estimator, X, y[:, i], eval_set=eval_sets_[i], **fit_params_validated
                    )
                    for i in range(y.shape[1])
                )
        else:
            self.estimators_ = Parallel(n_jobs=self.n_jobs)(
                delayed(_fit_estimator)(
                    self.estimator, X, y[:, i], **fit_params_validated
                )
                for i in range(y.shape[1])
            )

        if hasattr(self.estimators_[0], "n_features_in_"):
            self.n_features_in_ = self.estimators_[0].n_features_in_
        if hasattr(self.estimators_[0], "feature_names_in_"):
            self.feature_names_in_ = self.estimators_[0].feature_names_in_

        return self


class MultivariableMultiOutputRegressor(ForecastingMixin):
    """多变量多输出预测"""

    def __init__(self):
        pass

    def check_x_y(self, x, y):
        assert isinstance(x, (np.ndarray, torch.Tensor)) and isinstance(y, (np.ndarray, torch.Tensor)), \
            "only accept numpy.ndarray or torch.Tensor"
        assert x.ndim == 3

    def fit(self, x, **kwargs):
        pass

    def predict(self, x, n, **kwargs):
        pass
