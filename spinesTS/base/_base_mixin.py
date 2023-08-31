import numpy as np
from spinesUtils.feature_tools import vars_threshold, variation_threshold

from spinesTS.metrics import r2_score


class ForecastingMixin:
    def extend_predict(self, x, n):
        """Extrapolation prediction.

        Parameters
        ----------
        x: to_predict data, must be 2 dims data
        n: predict steps, must be int

        Returns
        -------
        np.ndarray, which has 2 dims

        """
        assert isinstance(n, int)
        assert x.ndim == 2

        current_res = self.predict(x)

        if n is None:
            return current_res
        elif n <= current_res.shape[1]:
            return current_res[:, :n]
        else:
            res = [current_res]
            for i in range((n//res[0].shape[1])+1):
                current_res = self.predict(x)
                res.append(current_res)
                x = np.concatenate((x[:, current_res.shape[1]:], current_res), axis=-1)

            res = np.concatenate(res, axis=-1)[:, :n]

            return res

    def score(self, x, y, eval2d=True):
        if eval2d:
            return r2_score(y.T, self.predict(x).T)
        else:
            return r2_score(y, self.predict(x))


class TableFeatureGenerateMixin:
    """Table Feature Generate Mixin class"""
    def features_filter(self, x):
        to_remove_col_names = list(set(vars_threshold(x) + variation_threshold(x)))
        if len(to_remove_col_names) > 0:
            x.drop(columns=to_remove_col_names, inplace=True)

        return x
