from activation_functions import sigmoid, softmax, ReLu, tanh
from layer import Layer
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import pickle
import matplotlib.pyplot as plt
from copy import deepcopy
from sklearn.metrics import accuracy_score
from typing import List

class Perceptron:
	def __init__(self, layers=[8, 8], activation=['sigmoid', 'sigmoid'],  epochs=50,\
			  loss_function='binaryCrossentropy', lr=1e-4, batch_size=16, early_stop=False) -> None:
		self.layers = []
		for nb_neuron, acti in zip(layers, activation):
			self.layers.append(Layer(nb_neuron, acti))
		self.layers.append(Layer(2, "softmax"))
		self.epochs = epochs
		self.loss_function = loss_function
		self.batch_size = batch_size
		self.lr = lr
		self.early_stop = early_stop


	def backpropagation(self, X, y, caches):
		m = X.shape[0]
		dA = caches[-1][0] - y
		for i in  reversed(range(len(self.layers))):
			A, Z, W, b = caches[i]

			if self.layers[i].activation_f == softmax:
				dZ = dA
			else:
				dZ = dA * self.layers[i].derivative_f(Z)
			dW = np.dot(caches[i-1][0].T, dZ) / m if i > 0 else np.dot(X.T, dZ) / m
			db = np.sum(dZ, axis=0, keepdims=True) / m
			dA = np.dot(dZ, W.T)

			self.layers[i].update_weight_and_bias([dW, db], self.lr)


	def train(self, X, y, X_valid, y_valid):
		if self.early_stop:
			old_layers = []
		if self.layers[-1].activation_f == softmax:
			y, y_valid = self.one_hot_encoded(y, y_valid)
		else:
			y = y.reshape(-1, 1)
		loss = []
		acc = []
		acc2 = []
		for iter in range(self.epochs):
			A = X
			caches = []

			# FORWARD PROPAGATION
			for i in range(len(self.layers)):
				Z, A = self.layers[i].feedforwarding(A, self.batch_size)
				caches.append((A, Z, self.layers[i].W, self.layers[i].b))

			output = A
			# BACKWARD PROPAGATION
			self.backpropagation(X, y, caches)


			if iter % 25 == 0:
				A_valid = X_valid
				for i in range(len(self.layers)):
					_, A_valid = self.layers[i].feedforwarding(A_valid, self.batch_size)
				loss.append(self.loss_function(y, output))
				acc.append(accuracy_score(y, self.predict(output)))
				acc2.append(accuracy_score(y_valid, self.predict(A_valid)))

				print(f'ITER : {iter:<4} LOSS : {loss[-1]:<20} ACCURACY :{acc[-1]:<20} VALIDATION_ACCURACY :{acc2[-1]:<20}', end='\r')
				if self.early_stop :
					if self.early_stop_check(acc, acc2):
						print(f'\nEarly stop before overfitting at iter = {iter}, last weights retablished')
						self.layers = deepcopy(old_layers[-1])
						break
					else:
						old_layers.append(deepcopy(self.layers))
				self.plot_history(loss, acc, acc2)
		plt.show()


	def early_stop_check(self, accuracy: List, validation_accuracy: List)-> bool:
		if len(validation_accuracy) >= 3:
			if (validation_accuracy[-2] > validation_accuracy[-1] and accuracy[-2] == accuracy[-1]) or\
				(validation_accuracy[-2] == validation_accuracy[-1] and accuracy[-2] == accuracy[-1]):
				return True
		else:
			return False


	def plot_history(self, loss: List, acc: List, acc2: List):
		plt.plot(loss, 'C0', label='Loss')
		plt.plot(acc, 'C1', label='Train acc')
		plt.plot(acc2, 'C2', label='Valid acc')
		if len(loss) == 1:
			plt.xlim(0, 25)
			plt.ylim(0, 1.5)
			plt.legend()
		elif len(loss) >= 25:
			plt.xlim(0, len(loss) + 5)
		plt.pause(0.1)


	def one_hot_encoded(self, y, y_valid):
		y_one_hot = np.zeros((y.size, y.max() + 1))
		y_one_hot[np.arange(y.size), y] = 1
		y = y_one_hot
		y_one_hot = np.zeros((y_valid.size, y_valid.max() + 1))
		y_one_hot[np.arange(y_valid.size), y_valid] = 1
		y_valid = y_one_hot
		return y, y_valid


	def predict(self, Z):
		res = Z >= 0.5
		return res


	def save_model(self, filename):
		with open(filename, 'wb') as f:
			pickle.dump(self.layers, f)


	def load_model(self, filename):
		with open(filename, 'rb') as f:
			self.layers = pickle.load(f)

