class Layer:
    def __init__(self, type_name, num_neurons, conv_size, activation_name):
        self.type_name = type_name
        self.num_neurons = num_neurons
        self.conv_size = conv_size
        self.activation_name = activation_name


class ModelAttributes:
    def __init__(self):
        self.id = -1
        self.optimizer_name = ''
        self.learning_rate = -1
        self.validation_split = -1
        self.num_epochs = -1
        self.batch_size = -1
        self.loss = -1
        self.accuracy = -1
        self.layers = []

    '''Returns a 1D array of strings'''
    def get_writable(self):
        writable = [self.id, self.optimizer_name, self.learning_rate, self.validation_split,
                    self.num_epochs, self.batch_size, self.loss, self.accuracy]

        for layer in self.layers:
            writable.append(layer.type_name)
            writable.append(layer.num_neurons)
            writable.append(layer.conv_size)
            writable.append(layer.activation_name)

        return writable

    '''Returns a 1D array of strings'''
    def get_header_writable(self):
        header = ['id', 'optimizer_name', 'learning_rate', 'validation_split', 'num_epochs',
                  'batch_size', 'loss', 'accuracy', 'layers-->']

        return header



# Parameters
# optimizer_name
# learning_rate
# validation split
# num_epochs
# batch_size
# num_layers
    # layer type
    # num_neurons
    # activation