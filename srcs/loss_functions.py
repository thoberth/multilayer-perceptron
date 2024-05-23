import numpy as np

def binarycrossentropy(y, output, eps=1e-32)-> float:
	m = y.shape[0]
	res= -(1/m) * np.sum(y*np.log(output+eps)+(1 - y)*np.log(1 - output + eps))
	return res