import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten

DENSE_MODEL_DIR = '/home/iyevenko/DenserNet/saved_models/dense-'
DENSERNET_MODEL_DIR = '/home/iyevenko/DenserNet/saved_models/even-densernet-'

def get_dense_model(d_model, num_layers, image_dims, first_layer=True):
    model = tf.keras.models.Sequential()
    if first_layer:
        model.add(Input(shape=image_dims))
        model.add(Flatten())

    for _ in range(num_layers):
        model.add(Dense(d_model, activation='relu'))

    model.add(Dense(10))

    return model


class DenserLayer(tf.keras.layers.Layer):

    def __init__(self, d_layer, num_layers, trainable=True, name=None, dtype=None, dynamic=False):
        super().__init__(trainable, name, dtype, dynamic)

        self.d_layer = d_layer
        self.num_layers = num_layers
        self.dense_layers = [
            Dense(d_layer, activation='relu')
            for _ in range(num_layers)
        ]

    def call(self, inputs, **kwargs):
        X = self.dense_layers[0](inputs)
        # activations = X

        for i in range(1, self.num_layers):
            # X = self.dense_layers[i](X)
            # activations = tf.keras.layers.concatenate([activations, X], -1)
            X = tf.keras.layers.concatenate([X, self.dense_layers[i](X)], -1)

        return X


def get_densernet(num_denser_layers, d_layer, num_layers, image_dims):
    model = tf.keras.Sequential()
    model.add(Input(shape=image_dims))
    model.add(Flatten())

    for i in range(num_denser_layers):
        model.add(DenserLayer(d_layer, num_layers))

    # model.add(get_dense_model(256, num_layers, image_dims, first_layer=False))
    model.add(get_dense_model(0, 0, image_dims, first_layer=False))

    return model


class WeightClipCallback(tf.keras.callbacks.Callback):

    def __init__(self, skip_thresh_factor, w_thresh_factor):
        super().__init__()
        self.skip_thresh_factor = skip_thresh_factor
        self.w_thresh_factor = w_thresh_factor

    def on_epoch_end(self, epoch, logs=None):
        denser_layers = self.model.layers[1:-1]
        for i in range(len(denser_layers)):
            w = denser_layers[i].get_weights()
            d_layer = denser_layers[0].d_layer
            for j in range(len(w)):
                if (i == 0 and j < 2):
                    continue
                elif j % 2 == 0:
                    wj = w[j][:d_layer,:]
                    wj_min, wj_max = 0, tf.reduce_max(tf.abs(wj))
                    w_thresh = self.w_thresh_factor * (wj_max-wj_min)
                    w_mask = tf.cast(tf.greater_equal(tf.abs(wj), w_thresh), tf.float32)

                    skip_w = w[j][d_layer:,:]
                    skip_min, skip_max = 0, tf.reduce_max(tf.abs(skip_w))
                    thresh = self.skip_thresh_factor * (skip_max - skip_min)
                    skip_mask = tf.cast(tf.greater(tf.abs(skip_w), thresh), tf.float32)

                    mask = tf.concat([w_mask, skip_mask], 0)
                    w[j] = w[j] * mask

            denser_layers[i].set_weights(w)