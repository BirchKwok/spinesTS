from spinesTS.metrics import r2_score
from spinesTS.utils import func_has_params


class MLModelMixin:
    def score(self, X, y, eval2d=True):
        if eval2d:
            return r2_score(y.T, self.predict(X).T)
        else:
            return r2_score(y, self.predict(X))

    def _fit(self, x, y, eval_set=None, **kwargs):
        if func_has_params(self._model.fit, 'eval_set') \
                and eval_set is not None:
            self._model.fit(x, y, eval_set=eval_set, **kwargs)
        else:
            if eval_set is not None:
                print(f"# function {self._model.fit.__qualname__} doesn't have eval_set parameter,"
                      f"the input will be ignored.")
            self._model.fit(x, y, **kwargs)

        return self._model



