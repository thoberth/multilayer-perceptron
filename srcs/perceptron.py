from layer import Layer
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

class Perceptron:
	def __init__(self, layers=[8, 8], epochs=50, loss_function='binaryCrossentropy', lr=1e-4, batch_size=16) -> None:
		self.layers = []
		for nb_neuron in layers:
			self.layers.append(Layer(nb_neuron))
		self.layers.append(Layer(1))
		self.epochs = epochs
		self.loss_function = loss_function
		self.batch_size = batch_size
		self.lr = lr


	def compute_gradients(self, X, y, Z):
		y = y.reshape(-1, 1).T
		gradients = []
		m = y.shape[1]
		dZ = Z[len(Z) - 1] - y
		for i in range(len(Z) - 1, -1, -1):
			if i == 0:
				dW = 1/m * dZ.dot(X.T)
			else :
				dW = 1/m * dZ.dot(Z[i - 1].T)
			db = 1/m * np.sum(dZ, axis=1, keepdims=True)
			if i!=0:
				dZ = np.dot(self.layers[i].W.T, dZ) * Z[i - 1] * (1 - Z[i - 1])
			gradients.append([dW, db])
		return gradients

	def backpropagation(self):
		pass


	def train(self, X, y, activation_function, X_valid, y_valid):
		loss = []
		acc = []
		acc2 = []

		for iter in range(self.epochs):
			Z = []
			# FORWARD PROPAGATION
			Z.append(activation_function(self.layers[0].feedforwarding(X.T, self.batch_size)))
			for i in range(1, len(self.layers)):
				Z.append(activation_function(self.layers[i].feedforwarding(Z[i-1], self.batch_size)))
			# BACKWARD PROPAGATION
			gradients = self.compute_gradients(X.T, y.T, Z)
			for i in range(len(self.layers) -1, -1, -1):
				self.layers[i].update_weight_and_bias(gradients[abs(i - (len(self.layers) - 1))], self.lr)


			if len(loss) > 0 and self.loss_function(y, Z[len(Z) - 1]) > loss[len(loss) -1]:
				print('Early stop')
				break
			if iter % 250 == 0:
				A = []
				A.append(activation_function(self.layers[0].feedforwarding(X_valid.T, self.batch_size)))
				for i in range(1, len(self.layers)):
					A.append(activation_function(self.layers[i].feedforwarding(A[i-1], self.batch_size)))
				loss.append(self.loss_function(y, Z[len(Z) - 1]))
				acc.append(accuracy_score(y.flatten(), self.predict(Z[len(Z) - 1].flatten())))
				acc2.append(accuracy_score(y_valid.flatten(), self.predict(A[len(A) - 1].flatten())))
				print('LOSS : ', loss[len(loss) - 1], '\tACCURACY', acc[len(acc) -1])
		plt.plot(loss)
		plt.plot(acc)
		plt.plot(acc2)
		plt.show()


	def predict(self, Z):
		res = Z >= 0.5
		# print(res)
		return res


	def save_model(self, filename):
		with open(filename, 'wb') as f:
			pickle.dump(self.weights, f)


	def load_model(self, filename):
		with open(filename, 'rb') as f:
			self.weights = pickle.load(f)