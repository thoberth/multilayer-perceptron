import numpy as np

def binarycrossentropy(y, y_predicted)-> float:
	print(y.shape, y_predicted.shape)
	res= (1/y.shape[0]) * sum([y[i]*np.log(y_predicted[i])+(1 - y[i])*np.log(1-y_predicted[i]) for i in range(y.shape[0])])
	res = np.where(res < 0.5, 0, 1)
	print("Resultat = ", res)
	return res