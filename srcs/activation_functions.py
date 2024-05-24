import numpy as np
import sklearn

def softmax(Z):
	exp_x = np.exp(Z - np.max(Z, axis=1, keepdims=True))
	return exp_x / np.sum(exp_x, axis=1, keepdims=True)

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
