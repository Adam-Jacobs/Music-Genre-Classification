import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import optimizers
from keras.utils import to_categorical
import random
import model_attributes as MA

# TODO - Add functionality for dynamic maximums based on memory allowance
# TODO - (for RL NN) Add functionality to take into account training time as a reward-To learn about diminishing returns
# TODO - expose only relevant methods to an outside script
# TODO - Add Dropout layer babyyy

# Increment Types
# incremental1 = [1, 2, 3, 4, 5]        # num_val=5, total=15 , av=3
# incremental2 = [1, 2, 3, 4, 5, 6]     # num_val=6, total=21 , av=3.5
# incremental3 = [1, 2, 3, 4, 5, 6, 7]  # num_val=7, total=28 , av=4

# fibonacci1 = [1, 2, 3, 5, 8]          # num_val=5, total=19 , av=3.8
# fibonacci2 = [1, 2, 3, 5, 8, 13]      # num_val=6, total=32 , av=5.33
# fibonacci3 = [1, 2, 3, 5, 8, 13, 21]  # num_val=7, total=53 , av=7.57

# binary1 = [1, 2, 4, 8, 16]            # num_val=5, total=31 , av=6.2
# binary2 = [1, 2, 4, 8, 16, 32]        # num_val=6, total=63 , av=10.5
# binary3 = [1, 2, 4, 8, 16, 32, 64]    # num_val=7, total=127, av=18.143

model_attributes = MA.ModelAttributes()

# Categorical Types
layer_types = ['Conv2D', 'MaxPooling2D', 'Dense']
activation_functions = ['softmax', 'elu', 'selu', 'softplus', 'softsign', 'relu',
                        'tanh', 'hard_sigmoid', 'exponential', 'linear',  '']
optimizer_names = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']


# Parameter Creation Functions
def get_activation_name():
    return activation_functions[random.randint(0, len(activation_functions) - 1)]


def get_activation_layer():
    activation_name = get_activation_name()
    if activation_name != '':
        return Activation(activation_name), activation_name
    else:
        return None, ''


def get_learning_rate():
    learning_rate_min = 0.001
    learning_rate_max = 1.0

    learning_rate = random.uniform(learning_rate_min, learning_rate_max)

    model_attributes.learning_rate = learning_rate

    return learning_rate


def get_optimizer():
    optimizer_name = optimizer_names[random.randint(0, len(optimizer_names) - 1)]
    model_attributes.optimizer_name = optimizer_name

    if optimizer_name == 'SGD':
        return optimizers.SGD(lr=get_learning_rate())
    elif optimizer_name == 'RMSprop':
        return optimizers.RMSprop(lr=get_learning_rate())
    elif optimizer_name == 'Adagrad':
        return optimizers.Adagrad(lr=get_learning_rate())
    elif optimizer_name == 'Adadelta':
        return optimizers.Adadelta(lr=get_learning_rate())
    elif optimizer_name == 'Adam':
        return optimizers.Adam(lr=get_learning_rate())
    elif optimizer_name == 'Adamax':
        return optimizers.Adamax(lr=get_learning_rate())
    elif optimizer_name == 'Nadam':
        return optimizers.Nadam(lr=get_learning_rate())

    return None


def get_batch_size():
    batch_size_min = 1
    batch_size_max = 25  # Because Memory

    batch_size = random.randint(batch_size_min, batch_size_max)

    model_attributes.batch_size = batch_size

    return batch_size


def get_num_epochs():
    num_epochs_min = 1
    num_epochs_max = 15  # Because Time

    num_epochs = random.randint(num_epochs_min, num_epochs_max)

    model_attributes.num_epochs = num_epochs

    return num_epochs


def get_validation_split_num():
    validation_split_min = 0.0
    validation_split_max = 0.5

    validation_split = random.uniform(validation_split_min, validation_split_max)
    # Make it more likely to have a lower validation split
    validation_split = random.uniform(validation_split_min, validation_split)

    model_attributes.validation_split = validation_split

    return validation_split


def get_num_layers():
    layer_min = 0
    layer_max = 15  # Because Memory

    return random.randint(layer_min, layer_max)


def get_num_neurons():
    neuron_min = 1
    neuron_max = 300  # Because Memory

    return random.randint(neuron_min, neuron_max)


def get_conv_shape():
    conv_shape_min = 2  # TODO - make this random based on the shape of the previous output layer?
    conv_shape_max = 5  # max theoretically just the size of the image(per dimension)?-no max is previous output shape

    size = random.randint(conv_shape_min, conv_shape_max)
    shape = (size, size)

    return shape, size


def get_conv2d_layer(is_input, input_shape):
    num_neurons = get_num_neurons()
    conv_shape, conv_size = get_conv_shape()
    if is_input:
        return Conv2D(num_neurons, conv_shape, input_shape=input_shape), num_neurons, conv_size
    else:
        return Conv2D(num_neurons, conv_shape), num_neurons, conv_size


def get_maxpooling2d_layer():
    conv_shape, conv_size = get_conv_shape()
    return MaxPooling2D(pool_size=conv_shape), 0, conv_size


def get_dense_layer():
    num_neurons = get_num_neurons()
    return Dense(num_neurons), num_neurons, 0


def get_layer(is_input, input_shape):
    if is_input:
        layer_type = 'Conv2D'
        layer, num_neurons, conv_size = get_conv2d_layer(True, input_shape)
    else:
        layer_type = layer_types[random.randint(0, len(layer_types) - 1)]
        if layer_type == 'Conv2D':
            layer, num_neurons, conv_size = get_conv2d_layer(False, None)
        elif layer_type == 'MaxPooling2D':
            layer, num_neurons, conv_size = get_maxpooling2d_layer()
        elif layer_type == 'Dense':
            layer, num_neurons, conv_size = get_dense_layer()

    activation, activation_name = get_activation_layer()

    model_attributes.layers.append(MA.Layer(layer_type, num_neurons, conv_size, activation_name))

    return layer, activation


def get_layers(input_shape):
    num_layers = get_num_layers()
    layers = []

    # Input Layer
    layer, activation = get_layer(True, input_shape)
    layers.append(layer)
    if activation is not None:
        layers.append(activation)

    # Hidden Layers
    for _ in range(num_layers):
        layer, activation = get_layer(False, None)
        layers.append(layer)
        if activation is not None:
            layers.append(activation)

    return layers


def reset():
    global model_attributes
    model_attributes = MA.ModelAttributes()

# Parameters
# optimizer
# learning_rate
# validation split
# num_epochs
# batch_size
# num_layers
    # layer type
    # num_neurons
    # activation
