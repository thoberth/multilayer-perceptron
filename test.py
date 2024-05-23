import numpy as np

# Activation functions
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x)**2

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def softmax_derivative(x):
    return x * (1 - x)

# Get activation function and its derivative
def get_activation_function(name):
    if name == "relu":
        return relu, relu_derivative
    elif name == "tanh":
        return tanh, tanh_derivative
    elif name == "softmax":
        return softmax, softmax_derivative
    else:
        raise ValueError(f"Unknown activation function: {name}")

# Neural Network class
class NeuralNetwork:
    def __init__(self, config):
        self.layers = []
        self.activations = []
        self.activation_derivatives = []
        
        input_size = config["input_size"]
        
        for layer in config["layers"]:
            layer_type = layer["type"]
            neurons = layer["neurons"]
            activation_name = layer["activation"]
            activation, activation_derivative = get_activation_function(activation_name)

            # Initialize weights and biases
            W = np.random.randn(input_size, neurons) * 0.01
            b = np.zeros((1, neurons))
            
            self.layers.append((W, b))
            self.activations.append(activation)
            self.activation_derivatives.append(activation_derivative)
            
            input_size = neurons
    
    def forward(self, X):
        A = X
        caches = []
        
        for (W, b), activation in zip(self.layers, self.activations):
            print('ICI', A.shape, W.shape)
            Z = np.dot(A, W) + b
            A = activation(Z)
            caches.append((A, Z, W, b))
        
        return A, caches
    
    def backward(self, X, y_true, caches):
        m = X.shape[0]
        grads = []
        dA = caches[-1][0] - y_true  # Derivative of loss with respect to output
        
        for i in reversed(range(len(self.layers))):
            A, Z, W, b = caches[i]
            activation_derivative = self.activation_derivatives[i]
            
            dZ = dA * activation_derivative(Z)
            dW = np.dot(caches[i-1][0].T, dZ) / m if i > 0 else np.dot(X.T, dZ) / m
            db = np.sum(dZ, axis=0, keepdims=True) / m
            dA = np.dot(dZ, W.T)
            
            grads.insert(0, (dW, db))
        
        return grads
    
    def update_parameters(self, grads, learning_rate):
        for i in range(len(self.layers)):
            W, b = self.layers[i]
            dW, db = grads[i]
            
            W -= learning_rate * dW
            b -= learning_rate * db
            
            self.layers[i] = (W, b)

# Example usage
import json

config_json = '''{
    "input_size": 426,
    "layers": [
        {
            "type": "dense",
            "neurons": 5,
            "activation": "relu"
        },
        {
            "type": "dense",
            "neurons": 3,
            "activation": "tanh"
        },
        {
            "type": "dense",
            "neurons": 2,
            "activation": "softmax"
        }
    ]
}'''

config = json.loads(config_json)

nn = NeuralNetwork(config)
X = np.random.randn(3, config["input_size"])  # Example input data
y_true = np.array([[1, 0], [0, 1],[1, 0]])  # Example true labels (one-hot encoded)

# Forward pass
print(X.shape, y_true.shape, X, '\n', y_true)
output, caches = nn.forward(X)

# Compute loss (cross-entropy loss for softmax output)
loss = -np.sum(y_true * np.log(output)) / y_true.shape[0]
print(f'Loss: {loss}')

# Backward pass
grads = nn.backward(X, y_true, caches)

# Update parameters
learning_rate = 0.01
nn.update_parameters(grads, learning_rate)
