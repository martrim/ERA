from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Conv2D, BatchNormalization, Flatten, Dense
from tensorflow.keras.layers import AveragePooling2D, MaxPooling2D, Activation, Softmax
from Example_Utilities.Activations import get_activation, get_normalization
from ERA import get_rational_parameters


def get_network(era_args):
    def add_VGG_block(n_filters, filter_size=(3, 3), n_conv=2, input=False, maxpooling=True, avpooling=False,
                      batchnorm=False, strides=1):
        if input:
            network.add(Conv2D(n_filters, filter_size, padding='same', kernel_initializer=kernel_initializer,
                             input_shape=(28, 28, 1), kernel_regularizer=l2(0.001)))
            network.add(get_normalization(model=network, standardization_axes=[1, 2], learnable_axes=[3]))
            network.add(get_activation(numerator, denominator, era_args, model=network))
            no_remaining_convolutions = n_conv - 1
        else:
            network.add(Conv2D(n_filters, filter_size, strides=strides, padding='same',
                             kernel_initializer=kernel_initializer,
                             kernel_regularizer=l2(0.001)))
            network.add(get_normalization(model=network, standardization_axes=[1, 2], learnable_axes=[3]))
            network.add(get_activation(numerator, denominator, era_args, model=network))
            no_remaining_convolutions = n_conv
        for _ in range(no_remaining_convolutions):
            network.add(Conv2D(n_filters, filter_size, padding='same', kernel_initializer=kernel_initializer,
                             kernel_regularizer=l2(0.001)))
            network.add(get_normalization(model=network, standardization_axes=[1, 2], learnable_axes=[3]))
            network.add(get_activation(numerator, denominator, era_args, model=network))
            if batchnorm:
                network.add(BatchNormalization())
        if batchnorm:
            network.add(BatchNormalization())
        if strides > 1:
            avpooling = maxpooling = False
        if avpooling:
            maxpooling = False
            network.add(AveragePooling2D((2, 2)))
        if maxpooling:
            network.add(MaxPooling2D((2, 2), padding='same'))

    kernel_initializer = 'glorot_uniform'
    num_labels = 10
    if era_args.initialization != 'random':
        numerator, denominator = get_rational_parameters(era_args)
    else:
        numerator = denominator = 0
    network = Sequential()
    add_VGG_block(64, input=True)
    network.add(Flatten())
    network.add(Dense(128, kernel_initializer=kernel_initializer, kernel_regularizer=l2(0.001)))
    network.add(get_normalization(model=network, standardization_axes=[1], learnable_axes=[]))
    network.add(get_activation(numerator, denominator, era_args, model=network))
    network.add(Dense(num_labels, kernel_initializer=kernel_initializer, kernel_regularizer=l2(0.001)))
    network.add(Activation(Softmax()))
    return network
