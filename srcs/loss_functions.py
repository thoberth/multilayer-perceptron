import numpy as np

def binarycrossentropy(y, Z, eps=1e-32)-> float:
	m = y.shape[0]
	res= -(1/m) * np.sum(y*np.log(Z+eps)+(1 - y)*np.log(1 - Z + eps))
	return res