from layer import Layer
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import pickle

class Perceptron:
	def __init__(self, layers=[8, 8], epochs=50, loss_function='binaryCrossentropy', lr=1e-4, batch_size=16) -> None:
		self.layers = []
		for nb_neuron in layers:
			self.layers.append(Layer(nb_neuron))
		self.epochs = epochs
		self.loss_function = loss_function
		self.batch_size = batch_size
		self.lr = lr


	def backpropagation(self):
		pass


	def train(self, X, y, activation_function, loss_function):
		for _ in range(self.epochs):
			for layer in self.layers:
				Z = activation_function(layer.feedforwarding(X, self.batch_size))
			y_predict = loss_function(y, Z)
			y_predict
			# ici afficher l'avance en fonction du training et de validation dataset
			# print result // add them to a list to plot their training curves


	def predict(self):
		pass


	def save_model(self, filename):
		with open(filename, 'wb') as f:
			pickle.dump(self.weights, f)


	def load_model(self, filename):
		with open(filename, 'rb') as f:
			self.weights = pickle.load(f)