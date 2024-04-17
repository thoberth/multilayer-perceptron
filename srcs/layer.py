import numpy as np
from activation_functions  import sigmoid, softmax


class Layer:
	def __init__(self, neuron=8, activation_f=softmax) -> None:
		self.W = None
		self.b = None
		self.activation_f = activation_f
		self.nbr_neuron = neuron
		print(f"Creation d'un hidden layer avec {neuron} neurones...")


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
