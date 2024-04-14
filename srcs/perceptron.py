from layer import Layer
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import pickle
import matplotlib.pyplot as plt

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


	def backpropagation(self):
		pass


	def train(self, X, y, activation_function, loss_function):
		loss = []
		# print('X = |', X.shape)
		# Z = activation_function(self.layers[0].feedforwarding(X, self.batch_size))
		# print('Z = |', Z.shape)
		# for i in range(1, len(self.layers)):
		# 		Z = activation_function(self.layers[i].feedforwarding(Z, self.batch_size))
		for iter in range(self.epochs):
			Z = []
			# FORWARD PROPAGATION
			if len(Z) == 0 :
				Z.append(activation_function(self.layers[0].feedforwarding(X.T, self.batch_size)))
				for i in range(1, len(self.layers)):
					Z.append(activation_function(self.layers[i].feedforwarding(Z[i-1], self.batch_size)))
			print("sortie de la couche finale: ", Z[len(Z) - 1].shape)
			loss.append(loss_function(y, Z[len(Z) - 1]))
			# BACKWARD PROPAGATION
			# for layer in self.layers:
			# 	layer.update_weight_and_bias(X.T, y, Z, self.lr)
			# ici afficher l'avance en fonction du training et de validation dataset
			# print result // add them to a list to plot their training curves
		plt.plot([sum(i)/len(i) for i in loss])
		plt.show()


	def predict(self):
		pass


	def save_model(self, filename):
		with open(filename, 'wb') as f:
			pickle.dump(self.weights, f)


	def load_model(self, filename):
		with open(filename, 'rb') as f:
			self.weights = pickle.load(f)