import numpy as np

class Layer:
	def __init__(self, neuron=8) -> None:
		self.W = None
		self.b = None
		self.nbr_neuron = neuron
		print(f"Creation d'un hidden layer avec {neuron} neurones...")


	def update_weight_and_bias(self, X : np.ndarray, y : np.ndarray, Z : np.ndarray, lr: float):
		gradient_W = 1/y.shape[0] * np.sum(Z - y)
		gradient_b = 1/y.shape[0] * np.sum(Z - y)
		self.W = self.W - lr * gradient_W
		self.b = self.b - lr * gradient_b

	def feedforwarding(self, X : np.ndarray, batch: int):
		if not isinstance(self.W, np.ndarray):
			self.W = np.random.rand(self.nbr_neuron, X.shape[0])
			self.b = np.random.rand(self.nbr_neuron, 1)
		print('X shape =  ', X.shape, ' taille de W', self.W.shape, ' taille de b: ', self.b)
		x_batch = np.array_split(X, batch, axis=1)
		Z = []
		for batch in x_batch:
			Z.append(self.b + np.dot(self.W, batch)) # Z = W * X + b
		Z = np.concatenate(Z, axis = 1)
		print("taille de Z:", Z.shape)
		return Z