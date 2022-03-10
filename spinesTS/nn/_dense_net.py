from spinesTS.base._base_mixin import KerasModelMixin
import tensorflow as tf
from spinesTS.nn._tf_layers import AddAxis
from sklearn.base import RegressorMixin
from spinesTS.utils import seed_everything


class ConvBlock(tf.keras.layers.Layer):
    def __init__(self, num_channels):
        super(ConvBlock, self).__init__()
        self.bn = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()
        self.conv = tf.keras.layers.Conv1D(filters=num_channels,
                                           kernel_size=3, padding='causal')

        self.listLayers = [self.bn, self.relu, self.conv]

    def call(self, x):
        y = x
        for layer in self.listLayers.layers:
            y = layer(y)
        y = tf.keras.layers.concatenate([x, y], axis=-1)
        return y


class DenseBlock(tf.keras.layers.Layer):
    def __init__(self, num_convs, num_channels):
        super(DenseBlock, self).__init__()
        self.listLayers = []
        for _ in range(num_convs):
            self.listLayers.append(ConvBlock(num_channels))

    def call(self, x):
        for layer in self.listLayers.layers:
            x = layer(x)
        return x


class TransitionBlock(tf.keras.layers.Layer):
    def __init__(self, num_channels, **kwargs):
        super(TransitionBlock, self).__init__(**kwargs)
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()
        self.conv = tf.keras.layers.Conv1D(num_channels, kernel_size=1)
        self.avg_pool = tf.keras.layers.AvgPool1D(pool_size=2, padding='same')

    def call(self, x):
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.conv(x)
        return self.avg_pool(x)


def block_1(shape):
    return tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=shape),
        AddAxis(axis=-1),
        tf.keras.layers.Conv1D(64, kernel_size=3, strides=1, padding='causal'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPool1D(pool_size=2, padding='same')])


def block_2(shape):
    model = block_1(shape)
    # `num_channels`为当前的通道数
    num_channels, growth_rate = 64, 32
    num_convs_in_dense_blocks = [4, 4, 4, 4]

    for i, num_convs in enumerate(num_convs_in_dense_blocks):
        model.add(DenseBlock(num_convs, growth_rate))
        # 上一个稠密块的输出通道数
        num_channels += num_convs * growth_rate
        # 在稠密块之间添加一个转换层，使通道数量减半
        if i != len(num_convs_in_dense_blocks) - 1:
            num_channels = num_channels // 2 if num_channels // 2 != 0 else 1
            model.add(TransitionBlock(num_channels))
    return model


class DenseNet1DTFModel(RegressorMixin, KerasModelMixin):
    """
    spinesTS DenseNet1D tensorflow-model
    """
    def __init__(self, output_nums, learning_rate=0.1,
                 monitor='val_loss', min_delta=0, patience=10, restore_best_weights=True, random_seed=0):
        seed_everything(random_seed, tf_seed=True)
        super().__init__()
        self._output_nums = output_nums
        self.learning_rate = learning_rate
        self.model = None
        self._callback(monitor=monitor, min_delta=min_delta,
                       patience=patience, restore_best_weights=restore_best_weights)

    def _call(self):
        tf.keras.backend.clear_session()
        model = block_2(self._shape)
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.ReLU())
        model.add(tf.keras.layers.GlobalAvgPool1D())
        model.add(tf.keras.layers.Flatten())

        model.add(tf.keras.layers.Dense(2048))
        model.add(tf.keras.layers.ReLU())
        model.add(tf.keras.layers.Dropout(0.1))
        model.add(tf.keras.layers.Dense(1024))
        model.add(tf.keras.layers.ReLU())
        model.add(tf.keras.layers.Dropout(0.1))

        model.add(tf.keras.layers.Dense(self._output_nums))

        model.compile(
            metrics=['mae'], optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss=tf.keras.losses.MSE  # MAE
        )
        return model

    def fit(self, x, y, eval_set=None, epochs=1000, verbose=2, batch_size='auto', callbacks=None):
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
