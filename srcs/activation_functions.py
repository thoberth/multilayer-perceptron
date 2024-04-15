import numpy as np
import sklearn
# Softmax MANDATORY
# hyperboloid tangent
# rectified linear unit


def softmax(Z):
	# print(Z)
	res = 1/(1 + np.exp(- Z))
	# print(res)
	# res = np.exp(Z)/np.sum(np.exp(Z))
	return res

def sigmoid(Z):
	res = 1/(1 + np.exp(- Z))
	return res