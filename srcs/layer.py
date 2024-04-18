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
		print(f"Creation d'un hidden layer avec {neuron} neurones...")


	# def compute_grad(self, Input, output, y):
		

	def update_weight_and_bias(self, gradients, lr: float):
		self.W = self.W - lr * gradients[0]
		self.b = self.b - lr * gradients[1]


	def feedforwarding(self, X : np.ndarray, batch: int):
		if not isinstance(self.W, np.ndarray):
			np.random.seed(42)
			self.W = np.random.randn(self.nbr_neuron, X.shape[0])
			self.b = np.random.randn(self.nbr_neuron, 1)
		x_batch = np.array_split(X, batch, axis=1)
		Z = []
		for batch in x_batch:
			Z.append(self.W.dot(batch) + self.b) # Z = W * X + b
		Z = np.concatenate(Z, axis = 1)
		Z = self.activation_f(Z)
		return Z
