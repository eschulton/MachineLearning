import numpy as np


def relu(x):
    return np.maximum(0, x)


def relu_prime(x):
    return (x > 0) * 1.0


def lrelu(x):
    return np.where(x > 0, x, 0.01 * x)


def lrelu_prime(x):
    return np.where(x > 0, 1.0, 0.01)


def sigmoid(x):
    s = 1.0 / (1.0 + np.exp(-x))
    return np.minimum(0.999999999999999, s)


def sigmoid_prime(x):
    return sigmoid(x) * (1.0 - sigmoid(x))


class Layer:

    def __init__(self, network, nodes, activation_name=None):
        
        self.network = network

        if len(network.layers) >= 1:
            self.previous_layer = network.layers[-1]
        else:
            self.previous_layer = None

        self.nodes = nodes
        self.activation_name = None
        self.activation = None
        self.activation_prime = None

        self.W = None
        self.b = None
        self.Z = None
        self.A = None

        self.dW = None
        self.db = None
        self.dZ = None
        self.dA = None

        if self.previous_layer is None:
            return
        
        if activation_name == 'relu':
            self.activation = relu
            self.activation_prime = relu_prime
        elif activation_name == 'lrelu':
            self.activation = lrelu
            self.activation_prime = lrelu_prime
        elif activation_name == 'sigmoid':
            self.activation = sigmoid
            self.activation_prime = sigmoid_prime
        else:
            raise NotImplementedError

        self.initialize_weights()


    def initialize_weights(self):
        if self.previous_layer is not None:
            self.W = np.random.randn(self.nodes, self.previous_layer.nodes) * np.sqrt(1.0 / self.previous_layer.nodes) * self.network.initialization_factor
            self.b = np.zeros((self.nodes, 1))


    def forward_propagation(self):
        self.Z = np.dot(self.W, self.previous_layer.A) + self.b
        self.A = self.activation(self.Z)


    def backward_propagation(self):
        if self.previous_layer is None:
            return
        else:
            self.dZ = self.dA * self.activation_prime(self.Z)
            m = self.previous_layer.A.shape[1]
            self.dW = (1.0 / m) * np.dot(self.dZ, self.previous_layer.A.T)
            self.db = (1.0 / m) * np.sum(self.dZ, axis=1, keepdims=True)
            if self.previous_layer.previous_layer is not None:
                self.previous_layer.dA = np.dot(self.W.T, self.dZ)
    

    def update_weights(self):
        self.W -= self.network.learning_rate * self.dW
        self.b -= self.network.learning_rate * self.db


class Network:

    def __init__(self, num_inputs, learning_rate=1.0, initialization_factor=1.0):
        
        self.layers = []
        self.num_inputs = num_inputs
        self.learning_rate = learning_rate
        self.initialization_factor = initialization_factor
        self.cost = None
        self.cost_history = []
        self.gradients = None
        self.approximations = None

        self.layers.append(Layer(self, num_inputs))


    def __str__(self):
        s = f"Network ; num_inputs = {self.num_inputs} ; learning_rate = {self.learning_rate} ; initialization_factor = {self.initialization_factor}\n"
        for layer in self.layers:
            s += f"..... Layer ; nodes = {layer.nodes} ; activation_name = {layer.activation_name}\n"
        return s


    def add_layer(self, nodes, activation_name):
        self.layers.append(Layer(self, nodes, activation_name))


    def initialize_weights(self):
        for layer in self.layers:
            layer.initialize_weights()
    

    def forward_propagation(self, X):
        for layer in self.layers:
            if layer.previous_layer is None:
                layer.A = X
            else:
                layer.forward_propagation()
        self.AL = self.layers[-1].A
        self.predictions = (self.AL > 0.5) * 1.0
    

    def calculate_cost(self, Y):
        self.Y = Y
        self.cost = (-1.0 / self.Y.shape[1]) * np.sum((self.Y * np.log(self.AL)) + ((1.0 - Y) * np.log(1.0 - self.AL)))
        self.cost_history.append(self.cost)
        self.accuracy = np.sum(self.predictions == self.Y) * 100.0 / self.Y.shape[1]
    

    def backward_propagation(self):
        self.layers[-1].dA = - ((self.Y / self.AL) - (1.0 - self.Y) / (1.0 - self.AL))
        for layer in reversed(self.layers):
            layer.backward_propagation()
    

    def update_weights(self):
        for layer in self.layers:
            if layer.previous_layer is not None:
                layer.update_weights()


    def get_weights(self):
        weights = np.array([])
        for layer in self.layers:
            if layer.previous_layer is not None:
                weights = np.concatenate((weights, layer.W.flatten(), layer.b.flatten()))
        return weights
    

    def get_gradients(self):
        gradients = np.array([])
        for layer in self.layers:
            if layer.previous_layer is not None:
                gradients = np.concatenate((gradients, layer.dW.flatten(), layer.db.flatten()))
        return gradients
    

    def put_weights(self, weights):
        i = 0
        for layer in self.layers:
            if layer.previous_layer is not None:
                j = i + layer.W.shape[0] * layer.W.shape[1]
                layer.W = weights[i:j].reshape(layer.W.shape[0], layer.W.shape[1])
                i, j = j, j + layer.b.shape[0]
                layer.b = weights[i:j].reshape(layer.b.shape[0], 1)
                i = j
    

    def grad_check(self, epsilon=0.0000001):

        X = np.ones(self.num_inputs).reshape(-1, 1)
        Y = np.ones(1).reshape(-1, 1)

        self.initialize_weights()
        weights_orig = self.get_weights()

        self.forward_propagation(X)
        self.calculate_cost(Y)
        self.backward_propagation()
        gradients = self.get_gradients()
        approximations = np.zeros(len(gradients))

        for i, weight in enumerate(weights_orig):

            weights = weights_orig.copy()
            weights[i] = weight + epsilon
            self.put_weights(weights)
            self.forward_propagation(X)
            self.calculate_cost(Y)
            J_plus = self.cost

            weights = weights_orig.copy()
            weights[i] = weight - epsilon
            self.put_weights(weights)
            self.forward_propagation(X)
            self.calculate_cost(Y)
            J_minus = self.cost

            approximations[i] = (J_plus - J_minus) / (2.0 * epsilon)
        
        difference = np.linalg.norm(gradients - approximations) / (np.linalg.norm(gradients) + np.linalg.norm(approximations))

        if difference < epsilon:
            print(f"grad check passed ; difference = {difference}")
        else:
            print(f"grad check failed ; difference = {difference}")
        
        self.gradients = gradients
        self.approximations = approximations
    

    def train(self, X, Y, num_iterations=1000, print_mod=100):

        self.initialize_weights()

        for i in range(1, num_iterations + 1):
            self.forward_propagation(X)
            self.calculate_cost(Y)
            self.backward_propagation()
            self.update_weights()
            if print_mod == 1 or i % print_mod == 1:
                print(f"iteration = {i} ; cost = {self.cost} ; accuracy = {self.accuracy}%")
    
    def test(self, X, Y):

        self.forward_propagation(X)
        self.calculate_cost(Y)
        print(f"cost = {self.cost} ; accuracy = {self.accuracy}%")