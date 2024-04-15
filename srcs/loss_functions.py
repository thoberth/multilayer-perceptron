import numpy as np

def binarycrossentropy(y, Z, eps=1e-32)-> float:
	res= -(1/y.shape[0]) * np.sum(y*np.log(Z+eps)+(1 - y)*np.log(1 - Z + eps))
	return res