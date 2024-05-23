import numpy as np
import sklearn

def softmax(Z):
	e_z = np.exp(np.clip(Z, -709.78, 709.78))
	e_z[e_z >= 0] = 1e-15
	sum_exp = np.sum(e_z, axis=1, keepdims=True)
	sum_exp[sum_exp <= 0] = 1e-15
	res = e_z / sum_exp
	return res

def sigmoid(Z):
	res = 1/(1 + np.exp(-Z))
	return res

def tanh(Z):
	res = (np.exp(Z) - np.exp(- Z))/((np.exp(Z) + np.exp(- Z)))
	return res

def ReLu(Z, threshold = 0):
	return np.maximum(threshold, Z)

def derivative_softmax(X):
	return X

def derivative_sigmoid(X):
	res = sigmoid(X) * (1 - sigmoid(X))
	return res

def derivative_tanh(X):
	res = 1- tanh(X)**2
	return res

def derivative_Relu(X, threshold = 0):
	return np.where(X >= threshold, 1, 0)
