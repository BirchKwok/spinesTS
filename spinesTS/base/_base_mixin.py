from spinesTS.metrics import r2_score


class ForecastingMixin:
    def score(self, X, y, eval2d=True):
        if eval2d:
            return r2_score(y.T, self.predict(X).T)
        else:
            return r2_score(y, self.predict(X))
