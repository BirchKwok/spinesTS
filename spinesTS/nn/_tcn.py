from spinesTS.base._base_mixin import KerasModelMixin
import tensorflow as tf
from spinesTS.nn._tf_layers import AddAxis
from tcn import TCN
from sklearn.base import RegressorMixin
from spinesTS.utils import seed_everything
from ._tf_layers import BidirectionalTFLayer


class TCN1DTFModel(RegressorMixin, KerasModelMixin):
    """
    spinesTS TCN1D tensorflow-model
    """

    def __init__(self, output_nums, learning_rate=0.1, nb_filters=128, kernel_size=4, nb_stacks=2,
                 dropout=0.2, tcn_blocks=1, monitor='val_loss', min_delta=0,
                 patience=10, restore_best_weights=True, random_seed=0):
        """
        TCN model
        :param nb_filters: The number of filters to use in the convolutional layers, Can be a list
        :param kernel_size: The size of the kernel to use in each convolutional layer
        :param nb_stacks: The number of stacks of residual blocks to use
        :param tcn_blocks: The number of stacks of tcn blocks to use
        """
        seed_everything(random_seed, tf_seed=True)
        self._output_nums = output_nums
        self.model = None
        self._learning_rate = learning_rate
        self._callback(monitor=monitor, min_delta=min_delta,
                       patience=patience, restore_best_weights=restore_best_weights)
        self._nb_filters, self._kernel_size, self._nb_stacks, self._tcn_blocks, self._dropout \
            = \
            nb_filters, kernel_size, nb_stacks, tcn_blocks, dropout

        self.tcn_params = {
            'nb_filters': self._nb_filters,
            'kernel_size': self._kernel_size,
            'use_batch_norm': True,
            'return_sequences': True,
            'nb_stacks': self._nb_stacks,
            'dropout_rate': self._dropout
        }

    def _call(self):
        net = tf.keras.Sequential([
            tf.keras.Input(shape=self._shape),
            AddAxis(axis=-1)
        ])
        if self._tcn_blocks == 1:
            net.add(
                TCN(
                    **self.tcn_params
                )
            )
        else:
            bidirectional_layers = self._tcn_blocks // 2

            if bidirectional_layers * 2 != self._tcn_blocks:
                for i in range(bidirectional_layers - 1):
                    net.add(
                        BidirectionalTFLayer(
                            TCN(
                                **self.tcn_params
                            )
                        )
                    )
                # use the last layer as backward_tcn_layer of BidirectionalTF
                net.add(
                    BidirectionalTFLayer(
                        BidirectionalTFLayer(
                            TCN(
                                **self.tcn_params
                            )
                        ),
                        TCN(
                            **self.tcn_params
                        )
                    )
                )
            else:
                for i in range(bidirectional_layers):
                    net.add(
                        BidirectionalTFLayer(
                            TCN(
                                **self.tcn_params
                            )
                        )
                    )
        net.add(tf.keras.layers.GlobalAveragePooling1D())
        net.add(tf.keras.layers.Dense(1024, activation='selu'))
        net.add(tf.keras.layers.Dense(self._output_nums))

        net.compile(metrics=['mae'], optimizer=tf.keras.optimizers.Adam(learning_rate=self._learning_rate)
                    , loss=tf.keras.losses.Huber()
                    )
        return net

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
