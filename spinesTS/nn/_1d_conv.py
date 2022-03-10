from sklearn.base import RegressorMixin
from spinesTS.base._base_mixin import KerasModelMixin
import tensorflow as tf
from spinesTS.utils import seed_everything


def _batch_layer(x_layer):
    x_layer = tf.keras.layers.BatchNormalization()(x_layer)
    x_layer = tf.keras.layers.Dropout(0.1)(x_layer)
    return x_layer


def _1d_conv_block(x_layer, neuron_num):
    x_layer = _batch_layer(x_layer)
    x_layer = tf.keras.layers.Conv1D(neuron_num, kernel_size=2, padding='same')(x_layer)

    return x_layer


class OneDimConvTFModel(RegressorMixin, KerasModelMixin):
    """
    spinesTS OneDimConv tensorflow-model
    """
    def __init__(self, output_nums, learning_rate=0.1,
                 monitor='val_loss', min_delta=0, patience=10, restore_best_weights=True, random_seed=0):
        seed_everything(random_seed, tf_seed=True)
        self._output_nums = output_nums
        self._learning_rate = learning_rate
        self.model = None
        self._callback(monitor=monitor, min_delta=min_delta,
                       patience=patience, restore_best_weights=restore_best_weights)

    def _call(self):
        input_init = tf.keras.layers.Input(shape=self._shape)
        input_1 = _batch_layer(input_init)
        input_1 = tf.keras.layers.Dense(1024, activation='relu')(input_1)
        input_1 = _batch_layer(input_1)
        input_1 = tf.keras.layers.Dense(4096, activation='relu')(input_1)
        input_1 = tf.keras.layers.Reshape((256, 16))(input_1)

        input_1 = _1d_conv_block(input_1, 512)

        input_1 = tf.keras.layers.AveragePooling1D(2)(input_1)

        input_1 = _1d_conv_block(input_1, 512)

        input_2 = _1d_conv_block(input_1, 512)
        input_2 = _1d_conv_block(input_2, 512)

        residual_connect = tf.keras.layers.Dot(axes=-1)([input_1, input_2])
        residual_connect = tf.keras.layers.Flatten()(residual_connect)

        output = _batch_layer(residual_connect)
        output = tf.keras.layers.Dense(256, activation='relu')(output)
        output = tf.keras.layers.Dense(self._output_nums)(output)

        model = tf.keras.Model(inputs=input_init, outputs=output)

        model.compile(metrics=['mae'], optimizer=tf.keras.optimizers.Adam(learning_rate=self._learning_rate)
                      , loss=tf.keras.losses.Huber())
        return model

    def fit(self, x, y, eval_set=None, epochs=1000, verbose=2, batch_size='auto', callbacks=None):
        tf.keras.backend.clear_session()
        self._fit_on_batch(x, batch_size=batch_size)
        self.model = self._call()
        callbacks = callbacks or self._callback
        self.model.fit(x, y, validation_data=eval_set, epochs=epochs,
                       verbose=verbose, batch_size=self._batch_size if batch_size == 'auto' else batch_size,
                       callbacks=callbacks
                       )

    def predict(self, x):
        assert self.model is not None, "model not fitted yet."
        return self.model.predict(x)
