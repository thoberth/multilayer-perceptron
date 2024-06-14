import numpy as np
from activation_functions  import sigmoid, softmax, tanh, ReLu, derivative_Relu, derivative_sigmoid, derivative_softmax, derivative_tanh

class Layer:
	def __init__(self, neuron=8, activation_f='softmax') -> None:
		self.W = None
		self.b = None
		if activation_f == 'softmax':
			self.activation_f = softmax
			self.derivative_f = derivative_softmax
		elif activation_f == 'sigmoid':
			self.activation_f = sigmoid
			self.derivative_f = derivative_sigmoid
		elif activation_f == 'tanh':
			self.activation_f = tanh
			self.derivative_f = derivative_tanh
		elif activation_f == 'ReLu' :
			self.activation_f = ReLu
			self.derivative_f = derivative_Relu
		self.nbr_neuron = neuron
		print(f"Creation d'un hidden layer avec {neuron} neurones \
utilisant {activation_f} comme fonction d'activation...")


	def update_weight_and_bias_momentum(self, gradients, lr: float, momentum : float = 0.9):
		self.V = (momentum * self.V) - (lr * gradients[0])
		self.V_bias = (momentum * self.V_bias) - (lr * gradients[1])
		self.W += self.V
		self.b += self.V_bias

	def update_weight_and_bias(self, gradients, lr: float):
		self.W = self.W - lr * gradients[0]
		self.b = self.b - lr * gradients[1]


	def feedforwarding(self, X : np.ndarray, batch: int = 1):
		if not isinstance(self.W, np.ndarray):
			np.random.seed(42)
			self.W = np.random.randn(X.shape[1], self.nbr_neuron)
			self.V = np.zeros_like(self.W)
			self.b = np.random.randn(1, self.nbr_neuron)
			self.V_bias = np.zeros_like(self.b)
		x_batch = np.array_split(X, batch, axis=0)
		Z = []
		for batch in x_batch:
			Z.append(np.dot(batch, self.W) + self.b) # Z = W * X + b
		Z = np.concatenate(Z, axis = 0)
		A = self.activation_f(Z)
		return Z, A
	