import copy
from typing import List
import numpy as np
from spinesTS.base import ForecastingMixin


class Pipeline(ForecastingMixin):
    """estimators pipeline """

    def __init__(self, steps: List[tuple]):
        """
        Demo:
            '''python
            from spinesTS.pipeline import Pipeline
            from spinesTS.preprocessing import split_array
            from spinesTS.data import LoadElectricDataSets
            from sklearn.preprocessing import StandardScaler
            from spinesTS.nn import TCN1D

            X_train, X_test, y_train, y_test =  LoadElectricDataSets().split_ds()

            pp = Pipeline([
                ('sc', 'StandardScaler()),
                ('tcn', 'TCN1D(30, 30))
            ])

            pp.fit(X_train, y_train)

            y_hat = pp.predict(X_test)

            print(pp.score(X_test, y_test))
            '''
        """
        assert 0 < len(steps) == np.sum([isinstance(i, tuple) for i in steps])

        self._names, self._estimators = zip(*steps)
        self._estimator = self._estimators[-1]
        # validate steps
        self._validate_steps()

        self._init_steps = steps
        self._order_steps = dict()
        for n, c in zip(self._names, self._estimators):
            self._order_steps[n] = c.__class__.__name__

    def fit(self, train_x, train_y, eval_set=None, **kwargs):
        x = copy.deepcopy(train_x)
        y = copy.deepcopy(train_y)
        for t in range(len(self._estimators[:-1])):
            if hasattr(t, 'fit_transform'):
                x = self._estimators[t].fit_transform(x)
            else:
                self._estimators[t].fit(x)
                x = self._estimators[t].transform(x)
            if eval_set is not None:
                _target = copy.deepcopy(eval_set)
                if isinstance(_target[0], tuple):
                    ex, ey = _target[0]
                    ex = self._estimators[t].transform(ex)
                    eval_set = [(ex, ey)]
                else:
                    ex, ey = _target
                    ex = self._estimators[t].transform(ex)
                    eval_set = (ex, ey)

        self._estimator.fit(x, y, eval_set=eval_set, **kwargs)

        return self

    def predict(self, x_pred, **kwargs):
        x = copy.deepcopy(x_pred)
        for t in range(len(self._estimators[:-1])):
            x = self._estimators[t].transform(x)

        return self._estimator.predict(x, **kwargs)

    def get_params(self):
        return copy.deepcopy(self._order_steps)

    def _validate_steps(self):

        transformers = self._estimators[:-1]
        estimator = self._estimator

        for t in transformers:
            if t is None:
                continue
            else:
                if not (hasattr(t, "fit") or hasattr(t, "fit_transform")) or not hasattr(
                        t, "transform"
                ):
                    raise TypeError(
                        "All intermediate steps should be "
                        "transformers and implement fit and transform "
                        "'%s' (type %s) doesn't" % (t, type(t))
                    )
        if (
                estimator is not None
                and not hasattr(estimator, "fit") and not hasattr(estimator, "predict")
        ):
            raise TypeError(
                "Last step of Pipeline should implement fit and predict"
                "'%s' (type %s) doesn't" % (estimator, type(estimator))
            )
