import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from tensorflow.keras import backend as K
from tensorflow.python.keras.utils import tf_inspect, generic_utils


class BidirectionalTFLayer(tf.keras.layers.Layer):
    def __init__(self, layer, backward_layer=None):
        super(BidirectionalTFLayer, self).__init__()
        self.forward_layer = self._recreate_layer_from_config(layer)

        if backward_layer is None:
            self.backward_layer = self._recreate_layer_from_config(layer)
        else:
            self.backward_layer = backward_layer
            # Keep the custom backward layer config, so that we can save it later. The
            # layer's name might be updated below with prefix 'backward_', and we want
            # to preserve the original config.
            self._backward_layer_config = generic_utils.serialize_keras_object(
                backward_layer)

        self.forward_layer._name = 'forward_' + self.forward_layer.name
        self.backward_layer._name = 'backward_' + self.backward_layer.name

    def __call__(self, inputs):
        inputs_1 = self.forward_layer(inputs)
        _ = K.reverse(inputs, axes=-1)
        inputs_2 = self.backward_layer(_)
        return tf.keras.layers.Concatenate(axis=-1)([inputs_1, inputs_2])

    @staticmethod
    def _recreate_layer_from_config(layer):
        config = layer.get_config()
        if 'custom_objects' in tf_inspect.getfullargspec(
                layer.__class__.from_config).args:
            custom_objects = {}
            cell = getattr(layer, 'cell', None)
            if cell is not None:
                custom_objects[cell.__class__.__name__] = cell.__class__
                # For StackedRNNCells
                stacked_cells = getattr(cell, 'cells', [])
                for c in stacked_cells:
                    custom_objects[c.__class__.__name__] = c.__class__
            return layer.__class__.from_config(config, custom_objects=custom_objects)
        else:
            return layer.__class__.from_config(config)


class AddAxis(tf.keras.layers.Layer):
    """ Add Axis layer. """

    def __init__(self, axis=-1):
        super(AddAxis, self).__init__()
        self.axis = axis

    def call(self, inputs):
        return tf.expand_dims(inputs, axis=self.axis)


class TimesNum(tf.keras.layers.Layer):
    """ times nums layer. """

    def __init__(self, level_digit):
        super(TimesNum, self).__init__()
        self.level_digit = level_digit

    def __call__(self, inputs):
        return inputs * self.level_digit


class Tanh(tf.keras.layers.Layer):
    """ Use tanh layer. """

    def __init__(self):
        super(Tanh, self).__init__()

    def call(self, inputs):
        return tf.math.tanh(inputs)


class Squeeze(tf.keras.layers.Layer):
    """ Use squeeze layer. """

    def __init__(self):
        super(Squeeze, self).__init__()

    def call(self, inputs):
        return tf.squeeze(inputs)

