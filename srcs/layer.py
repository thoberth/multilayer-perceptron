import numpy as np

class Layer:
	def __init__(self, neuron=8) -> None:
		self.W = None
		self.b = None
		self.nbr_neuron = neuron
		print(f"Creation d'un hidden layer avec {neuron} neurones...")


	def update_weight(self, y_predicted):
		pass


	def feedforwarding(self, X, batch):
		if not isinstance(self.W, np.ndarray):
			self.W = np.random.rand(X.shape[1], self.nbr_neuron)
			self.b = np.random.rand(1)
		x_batch = np.array_split(X, batch)
		Z = []
		for batch in x_batch:
			Z.append(self.b + np.dot(batch, self.W)) # Z = W * X + b
		Z = np.concatenate(Z, axis = 0)
		print("taille de Z:", Z.shape)
		return np.concatenate(Z, axis = 0)
