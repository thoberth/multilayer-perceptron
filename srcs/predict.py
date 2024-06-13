from perceptron import Perceptron
import pandas as pd
import numpy as np
import argparse
from utils import control_file

if __name__=="__main__":
	parser = argparse.ArgumentParser(description="This program predict a NN Perceptron\
									and then train with chosen args")

	parser.add_argument('-d','--dataset', default='Validation_Dataset.csv', type=control_file, action='store', help='The Dataset to evaluate')

	parser.add_argument('-l', '--load_weight', default="file.pickle", type=control_file, action='store', help='The Weights of model to load')

	args = parser.parse_args()

	df_valid = pd.read_csv(args.dataset)
	X_valid = df_valid.iloc[:, 2:].to_numpy()
	y_valid = np.array([ 0 if x == 'B' else 1 for x in df_valid.iloc[:, 1].tolist()])
	p = Perceptron.load_model(args.load_weight)
	output = p.predict_res(X_valid)
	y_valid, _ = p.one_hot_encoded(y_valid, np.array([1]))
	loss = p.loss_function(y_valid, output)
	print(f"La loss est de {loss:.8}")