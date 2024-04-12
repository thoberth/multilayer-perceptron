import numpy as np
# Softmax MANDATORY
# hyperboloid tangent
# rectified linear unit


def softmax(X):
	print("Entree softmax X = ?", X.shape)
	to_divid = sum(np.exp(X))
	res = np.array([(np.exp(i))/to_divid for i in X])
	print("shape apres fonction d'activation", res.shape)
	return res