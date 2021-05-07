import datetime
import os

import tensorflow as tf
from densernet.model import *
import matplotlib.pyplot as plt
import numpy as np

def plot_weights(model):
    layers = model.layers[1:-1]
    fig, axs = plt.subplots(len(layers), len(layers[0]._layers[0]))

    for i in range(len(layers)):
        densernet_layers = layers[i]._layers[0]
        for j in range(len(densernet_layers)):
            if i == 0 and j == 0:
                if len(layers) == 1:
                    axs[j].imshow(np.zeros((4, 4, 1)), cmap='Greys')
                    axs[j].axis('off')
                else:
                    axs[i, j].imshow(np.zeros((4, 4, 1)), cmap='Greys')
                    axs[i, j].axis('off')

            if len(layers) == 1:
                axs[j].imshow(np.abs(densernet_layers[j].kernel.numpy()), cmap='YlGnBu', interpolation='none')
                axs[j].axis('off')
            else:
                axs[i, j].imshow(np.abs(densernet_layers[j].kernel.numpy()), cmap='YlGnBu', interpolation='none')
                axs[i, j].axis('off')

    fig.tight_layout()
    plt.show()


def print_num_weights(model):
    tw = model.trainable_weights
    sum_tw = 0
    sum_nonzero = 0
    for w in tw:
        sum_tw += tf.size(w)
        sum_nonzero += tf.reduce_sum(tf.cast(tf.greater(w, 0), tf.int32))

    print(f'Number of trainable weights: {sum_tw}')
    print(f'Number of nonzero weights: {sum_nonzero}')


if __name__ == '__main__':
    # (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path='mnist.npz')
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_val, x_test = np.split(x_test, 2, 0)
    y_val, y_test = np.split(y_test, 2, 0)
    x_train = tf.cast(x_train / 255, dtype=tf.float32)
    y_train = tf.squeeze(y_train, -1)
    x_val = tf.cast(x_val / 255, dtype=tf.float32)
    y_val = tf.squeeze(y_val, -1)
    x_test = tf.cast(x_test / 255, dtype=tf.float32)
    y_test = tf.squeeze(y_test, -1)

    print((x_train.shape, y_train.shape))

    image_dims = x_train.shape[1:]

    epochs = 20
    batch_size = 128

    num_denser_layers = 4
    d_model = 128
    num_layers = 8

    # model_type = 'dense'
    model_type = 'densernet'

    if model_type == 'dense':
        model = get_dense_model(d_model, num_layers, image_dims)
    else:
        model = get_densernet(num_denser_layers, d_model, num_layers, image_dims)

    model.summary()
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
                  run_eagerly=False)

    log_dir = os.path.join('../logs', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tb_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
                        callbacks=[tb_callback, WeightClipCallback(0.2, 0.1)], validation_data=(x_val, y_val))

    plot_weights(model)
    print_num_weights(model)

    test_loss, test_acc = model.evaluate(x_test, y_test, batch_size)
    print(f'Test Loss: {test_loss}\nTest Accuracy: {test_acc}')

    if model_type == 'dense':
        tf.saved_model.save(model, DENSE_MODEL_DIR + f'{num_layers}x{d_model}')
    else:
        tf.saved_model.save(model, DENSERNET_MODEL_DIR + f'{num_denser_layers}x({num_layers}x{d_model})')