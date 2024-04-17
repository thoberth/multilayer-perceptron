import numpy as np
import sklearn
# Softmax MANDATORY
# hyperboloid tangent
# rectified linear unit


def softmax(Z):
	e_z = np.exp(Z)
	res = e_z/ np.sum(e_z)
	# print('softmax')
	return res

def sigmoid(Z):
	# print('sigmoid')
	res = 1/(1 + np.exp(-Z))
	return res

def tanh(Z):
	res = (np.exp(Z) - np.exp(- Z))/((np.exp(Z) + np.exp(- Z)))
	return res

def ReLu(Z, threshold = 0):
	return np.max(threshold, Z)