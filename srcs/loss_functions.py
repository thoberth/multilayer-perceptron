import numpy as np

def binarycrossentropy(y, Z, eps=1e-15)-> float:
	res= -(1/y.shape[0]) * sum(y*np.log(Z+eps)+(1 - y)*np.log(1-Z + eps))
	print("Resultat = ", res)
	return res