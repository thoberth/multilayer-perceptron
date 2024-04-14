import numpy as np
# Softmax MANDATORY
# hyperboloid tangent
# rectified linear unit


def softmax(Z):
	print("Entree softmax X = ?", Z.shape)
	to_divid = sum(np.exp(Z))
	res = np.array([(np.exp(i))/to_divid for i in Z])
	# res = np.where(res < 0.5, 0, 1)
	print("shape apres fonction d'activation", res.shape)
	return res