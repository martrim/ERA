import tensorflow as tf


class Normalization(tf.keras.layers.Layer):
    def __init__(self,
                 input_shape,
                 standardization_axes=None,
                 learnable_axes=None,
                 epsilon=0.001,
                 name=None):
        super(Normalization, self).__init__(name=name)

        self.standardization_axes = standardization_axes
        self.learnable_axes = learnable_axes
        self.epsilon = epsilon
        self.new_shape = input_shape

        weight_shape = []
        for i in range(len(self.new_shape)):
            if i in self.learnable_axes:
                weight_shape.append(self.new_shape[i])
            else:
                weight_shape.append(1)
        weight_shape = tuple(weight_shape)
        self.beta = self.add_weight(
            shape=weight_shape,
            name=self.name + '/beta',
            initializer=tf.keras.initializers.Zeros())
        self.gamma = self.add_weight(
            shape=weight_shape,
            name=self.name + '/gamma',
            initializer=tf.keras.initializers.Ones())

    def call(self, inputs):
        mu = tf.math.reduce_mean(inputs, self.standardization_axes, keepdims=True)
        sigma = tf.math.reduce_std(inputs, self.standardization_axes, keepdims=True)

        return self.beta + self.gamma * (inputs - mu) / sigma