from perceptron import Perceptron
import numpy as np
from activation_functions import softmax
import matplotlib.pyplot as plt
from typing import List
class Tester_activations(Perceptron):
	def __init__(self, layers=[8,8], epochs=1000, lr=1e-4, metrics = []):
		self.activations = ['sigmoid', 'ReLu', 'tanh', 'softmax']
		self.perceptrons=[]
		self.epochs=epochs
		for acti in self.activations:
			print("Creation d'un Perceptron:")
			self.perceptrons.append(Perceptron(layers=layers,\
									  activation=[acti for _ in range(len(layers))], lr=lr, metrics=metrics))
			print()

	def train(self, X, y, X_valid, y_valid):
		self.init_plot()
		y, y_valid = self.one_hot_encoded(y, y_valid)
		loss = [[] for _ in range(len(self.perceptrons))]
		for iter in range(1, self.epochs+1):
			for index_percep, percep in enumerate(self.perceptrons):
				A = X
				caches = []

				# FORWARD PROPAGATION
				for i in range(len(percep.layers)):
					Z, A = percep.layers[i].feedforwarding(A, percep.batch_size)
					caches.append((A, Z, percep.layers[i].W, percep.layers[i].b))

				output = A
				# BACKWARD PROPAGATION
				percep.backpropagation(X, y, caches)

				if iter % 25 == 0:
					A_valid = X_valid
					for i in range(len(percep.layers)):
						_, A_valid = percep.layers[i].feedforwarding(A_valid, percep.batch_size)
					loss[index_percep].append(percep.loss_function(y, output))
					y_hat = percep.predict(output)
					y_valid_hat = percep.predict(A_valid)
					percep.update_metrics(y_hat=y_hat, y_true=y, y_valid_hat=y_valid_hat, y_valid_true=y_valid)
			if iter % 25 == 0:
				print(f"ITER {iter:<4} LOSS {loss[0][-1]:.8}", end='\r')
				self.plot_history(loss)
		plt.show()
		self.put_metrics()


	def plot_history(self, loss):
		for index_perceptron in range(len(self.perceptrons)):
			col = index_perceptron % 2
			row = index_perceptron // 2
			self.axs[row, col].plot(loss[index_perceptron], 'C0', label='Loss')
			self.axs[row, col].plot(self.perceptrons[index_perceptron].metrics['accuracy'], 'C1', label='Train acc')
			if 'f1 score' in self.perceptrons[index_perceptron].metrics:
				self.axs[row, col].plot(self.perceptrons[index_perceptron].metrics['f1 score'], 'C3', label='F1_score')
			if 'recall' in self.perceptrons[index_perceptron].metrics:
				self.axs[row, col].plot(self.perceptrons[index_perceptron].metrics['recall'], 'C4', label='recall')
			if 'auc' in self.perceptrons[index_perceptron].metrics:
				self.axs[row, col].plot(self.perceptrons[index_perceptron].metrics['auc'], 'C5', label='auc')
			if len(loss[index_perceptron]) == 1:
				self.axs[row, col].legend()
			elif len(loss[index_perceptron]) >= 25:
				self.axs[row, col].set_xlim(0, len(loss[index_perceptron]) + 5)
			plt.pause(0.001)



	def init_plot(self):
		self.fig, self.axs = plt.subplots(2, 2, figsize=(10, 8))
		for i, name in enumerate(self.activations):
			row, col = divmod(i, 2)
			self.axs[row, col].set_title(name)
			self.axs[row, col].set_xlim(0, 25)
			self.axs[row, col].set_ylim(0, 1.5)
		plt.tight_layout()


	def put_metrics(self):
		for percep, name in zip(self.perceptrons, self.activations):
			print(name.upper(), ':', end=" ")
			for metrics, value in percep.metrics.items():
				print(metrics, '=', f"{value[-1]:.7}", end=', ')
			print()