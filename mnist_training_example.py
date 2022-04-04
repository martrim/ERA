import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist
from Example_Utilities.Callbacks import get_callbacks
from Example_Utilities.Network_Definer import get_network
from Example_Utilities.Parser import parse_era_arguments


def scale_pixels(x_train, x_test):
    return x_train.astype('float32') / 255, x_test.astype('float32') / 255


def make_tf_dataset(x, y):
    x = tf.data.Dataset.from_tensor_slices(x)
    y = tf.data.Dataset.from_tensor_slices(y)
    dataset = tf.data.Dataset.zip((x, y))
    dataset = dataset.shuffle(2048)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset


def lr_scheduler(epoch):
    lr = 1e-4
    if epoch > 60:
        lr *= 1e-6
    elif epoch > 50:
        lr *= 1e-5
    elif epoch > 40:
        lr *= 1e-4
    elif epoch > 30:
        lr *= 1e-3
    elif epoch > 20:
        lr *= 1e-2
    elif epoch > 10:
        lr *= 1e-1
    return lr


era_args = parse_era_arguments()
era_args.degree_denominator = int(era_args.degree_denominator)

# Loading and processing data
training_size = 60000
test_size = 10000
input_shape = (28, 28, 1)
num_labels = 10
batch_size = 32
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = np.expand_dims(x_train, axis=3)
x_test = np.expand_dims(x_test, axis=3)
x_train, x_test = scale_pixels(x_train, x_test)
y_train, y_test = np.squeeze(y_train), np.squeeze(y_test)
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)
training = make_tf_dataset(x_train, y_train)
test = make_tf_dataset(x_test, y_test)

# Setting up the network
loss_function = tf.keras.losses.CategoricalCrossentropy()  # default: average over batch, else (.fit): sum
opt = Adam(learning_rate=lr_scheduler(epoch=0))
accuracy_metric = tf.keras.metrics.CategoricalAccuracy()
network = get_network(era_args)
network.compile(optimizer=opt, loss=loss_function, metrics=[accuracy_metric], run_eagerly=False)

num_training_steps= np.floor(training_size / batch_size)
num_test_steps= np.floor(test_size / batch_size)
training_verbosity = 1
history = network.fit(training, batch_size=batch_size, steps_per_epoch=num_training_steps,
                      epochs=100, validation_data=test, validation_steps=num_test_steps,
                      callbacks=get_callbacks(), verbose=training_verbosity)
