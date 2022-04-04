from Example_Utilities.normalization import Normalization
from ERA import ERA

activation_counter = 0
normalization_counter = 0


def get_input_shape(model):
    return model.layers[-1].output_shape


def get_normalization(input_shape=None, model=None, standardization_axes=None, learnable_axes=None):
    global normalization_counter
    normalization_counter += 1

    if input_shape is None:
        input_shape = get_input_shape(model)

    return Normalization(input_shape, standardization_axes, learnable_axes)


def get_activation(numerator, denominator, era_args, input_shape=None, model=None):
    global activation_counter
    activation_counter += 1

    if input_shape is None:
        input_shape = get_input_shape(model)
    return ERA(numerator, denominator, era_args, input_shape=input_shape,
               name=('rational' + str(activation_counter - 1)))

