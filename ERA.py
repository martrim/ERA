"""Defines the Enhanced Rational Activation (ERA).
"""
import numpy as np
import tensorflow as tf
from scipy.optimize import minpack


def approximated_gelu(x):
    return tf.keras.activations.gelu(x, approximate=True)


def get_rational_parameters(era_args, lower_bound=-3, upper_bound=3):
    degree_denominator = era_args.degree_denominator
    initialization = era_args.initialization
    print('Generating initial parameters.')
    if initialization == 'leaky':
        target_function = tf.nn.leaky_relu
    elif initialization == 'relu':
        target_function = tf.keras.activations.relu
    elif initialization == 'swish':
        target_function = tf.keras.activations.swish
    elif initialization == 'gelu':
        target_function = approximated_gelu
    else:
        raise ValueError('Invalid target_name.')

    num_weights = 2 * degree_denominator + 2

    p0 = np.ones(num_weights, dtype='float32')
    x_size = 100000
    x = np.linspace(lower_bound, upper_bound, x_size, dtype='float32')
    y = target_function(x)

    result = minpack.least_squares(lambda weights: era_function(
        x, weights[degree_denominator:], weights[:degree_denominator]) - y, p0, jac='3-point', method='dogbox')
    # jac='2-point'
    # method='trf'
    fitted_weights = result['x'][:, np.newaxis]
    numerator = tf.constant(fitted_weights[degree_denominator:], dtype=tf.float32)
    denominator = tf.constant(fitted_weights[:degree_denominator], dtype=tf.float32)
    return numerator, denominator


def era_function(x, numerator_weights, denominator_weights):
    output = numerator_weights[0] * x + numerator_weights[1]
    numerator_weights = numerator_weights[2:]

    num_partial_fractions = numerator_weights.shape[0] // 2
    for i in range(num_partial_fractions):
        output += (numerator_weights[2 * i] * x + numerator_weights[2 * i + 1]) / \
                  ((x - denominator_weights[2 * i]) ** 2 + denominator_weights[2 * i + 1] ** 2)
    return output


class ERA(tf.keras.layers.Layer):
    def __init__(self,
                 numerator,
                 denominator,
                 era_args,
                 input_shape,
                 name=None):
        super(ERA, self).__init__(name=name)

        assert len(input_shape) in [2, 3, 4, 5]

        degree_denominator = era_args.degree_denominator
        initialization = era_args.initialization
        num_numerator_weights = degree_denominator + 2

        self.num_channels = input_shape[-1]
        weight_shape_ending = [1] * len(input_shape)
        self.numerator_weight_shape = [num_numerator_weights] + weight_shape_ending
        self.denominator_weight_shape = [degree_denominator] + weight_shape_ending
        if initialization != 'random':
            numerator = tf.repeat(numerator, weight_shape_ending[-1], axis=-1)
            denominator = tf.repeat(denominator, weight_shape_ending[-1], axis=-1)
            numerator = tf.reshape(numerator, self.numerator_weight_shape)
            denominator = tf.reshape(denominator, self.denominator_weight_shape)

        # Adding trainable weight vectors for numerator and denominator
        if initialization == 'random':
            num_initializer = tf.keras.initializers.GlorotUniform()
            denom_initializer = tf.keras.initializers.GlorotUniform()
        else:
            num_initializer = tf.keras.initializers.Constant(numerator)
            denom_initializer = tf.keras.initializers.Constant(denominator)
        self.numerator = self.add_weight(
            shape=self.numerator_weight_shape,
            name=self.name + '/w_numerator',
            initializer=num_initializer,)
        self.denominator = self.add_weight(
            shape=self.denominator_weight_shape,
            name=self.name + '/w_denominator',
            initializer=denom_initializer)

    def call(self, inputs):
        return era_function(inputs, self.numerator, self.denominator)
