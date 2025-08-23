import pseudomath as math
import pseudopillow as pillow

LEARNING_RATE = 0.0001

class DataLoader:
    def __init__(self, folder):
        self.folder = folder
        self.num = 0
        self.i = 1

    def __iter__(self):
        return self

    def __next__(self):
        if self.num > 9:
            raise StopIteration
        file = f'{self.folder}/{self.num}/{self.num}_{self.i}.bmp'
        result = (self.num, math.Matrix(pillow.read_bmp(file)).flatten())
        self.i += 1
        if self.i > 1000:
            self.i = 1
            self.num += 1
        return result

class Layer:
    def __init__(self, input_size: int, output_size: int, activation='null'):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = math.Matrix.rand((output_size, input_size))
        self.bias = math.Matrix.rand((output_size, 1))
        self.activation = None if activation == 'null' else getattr(math, activation)

        self.prev_inputs = math.Matrix.const(0, (input_size, 1))
        self.this_outputs = math.Matrix.const(0, (output_size, 1))

        # deltas for backprop
        self.delta_W = math.Matrix.like(0, self.weights)
        self.delta_b = math.Matrix.like(0, self.bias)
        self.delta = math.Matrix.like(0, self.bias)

    def forward(self, input_vector: math.Matrix):
        self.prev_inputs = input_vector
        output = self.weights @ input_vector + self.bias
        self.this_outputs = output
        if self.activation is not None:
            output = self.activation(output)
        return output

    def calc_deltas(self, delta: math.Matrix):
        self.delta = delta
        self.delta_W = delta @ self.prev_inputs.transpose()
        self.delta_b = delta

    def apply_deltas(self):
        self.weights -= self.delta_W * LEARNING_RATE
        self.bias -= self.delta_b * LEARNING_RATE

    def __str__(self):
        string = 'Layer'
        if self.activation is not None:
            string += f' ({self.activation.__name__})'
        return f'{string}: {self.input_size} -> {self.output_size}'

    def __repr__(self):
        return self.__str__()

class Network:
    def __init__(self, layers: list[Layer]):
        self.layers = layers
        self.layer_count = len(layers)
        self.cost = -1

    def forward(self, input_vector: math.Matrix, verbose=False):
        for i, layer in enumerate(self.layers):
            if verbose:
                print(f'Forward ({i}/{self.layer_count}):')
                print(layer)
                print('\n')
            input_vector = layer.forward(input_vector)
        return input_vector

    def backpropagate(self, predicted_vector: math.Matrix, actual_vector: math.Matrix, verbose=False):
        m, n = predicted_vector.dim()
        self.cost = 0
        for i in range(m):
            for j in range(n):
                self.cost += 0.5 * ((predicted_vector[i, j] - actual_vector[i, j]) ** 2)
           
        if verbose:
            print(f'Cost: {self.cost:.3f}')

        for i in range(self.layer_count - 1, -1, -1):
            layer = self.layers[i]
            if i == self.layer_count - 1:
                delta = predicted_vector - actual_vector
            else:
                next_layer = self.layers[i + 1]
                delta = (next_layer.weights.transpose() @ next_layer.delta) * math.relu_prime(layer.this_outputs)
            layer.calc_deltas(delta)

        for layer in self.layers:
            layer.apply_deltas()

        if verbose:
            print('Backpropagation complete.\n')

    def __getitem__(self, item):
        return self.layers[item]

    def __str__(self):
        string = f'Neural Network ({self.layer_count} layers) {self.layers[0].input_size} -> {self.layers[-1].output_size}:\n----------------------\n'
        for layer in self.layers:
            string += str(layer) + '\n'
        return string

    def __repr__(self):
        return self.__str__()
