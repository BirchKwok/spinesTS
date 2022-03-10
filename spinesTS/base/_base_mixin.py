import numpy as np
from tensorflow import keras
from spinesTS.utils import func_has_params


class KerasModelMixin:
    def _callback(self, monitor='val_loss', min_delta=0, patience=10, restore_best_weights=True):
        self._callback = keras.callbacks.EarlyStopping(monitor=monitor, min_delta=min_delta,
                                                       patience=patience, restore_best_weights=restore_best_weights)

    def _get_batch_size(self, x, batch_size='auto'):
        if batch_size == 'auto':
            self._batch_size = 32 if len(x) < 800 else len(x) // 40
        else:
            assert isinstance(batch_size, int) and batch_size > 0
            self._batch_size = batch_size

    def _get_input_shape(self, x):
        assert isinstance(x, np.ndarray)
        self._shape = x.shape[-1]

    def _fit_on_batch(self, x, batch_size='auto'):
        self._get_batch_size(x, batch_size)
        self._get_input_shape(x)


class EstimatorMixin:
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



