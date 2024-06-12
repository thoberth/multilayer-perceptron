from tester_activations import Tester_activations
import pandas as pd
import numpy as np
import argparse
from utils import control_layers, control_epochs, control_lr, validate_metrics

if __name__=='__main__':
	parser = argparse.ArgumentParser(description="This create a NN Perceptron\
									and then train with chosen args")
	
	parser.add_argument('-e', '--epochs', type=control_epochs, default=1000, action='store', help='Number of epochs while training')

	parser.add_argument('-hl', '--layer', nargs='+', default=[8, 8], type=control_layers, help='Size of number of layers')

	parser.add_argument('-lr', '--learning_rate', default=0.01, type=control_lr, action='store', help='The value of the learning rate')

	parser.add_argument('--metrics', nargs='+', default=[], type=validate_metrics, help="Specify one or more metrics. Valid options are 'f1_Score', 'accuracy', 'recall'")

	args = parser.parse_args()

	tester = Tester_activations(layers=args.layer, epochs=args.epochs, lr=args.learning_rate, metrics=args.metrics)

	df = pd.read_csv('Training_Dataset.csv')
	df_valid = pd.read_csv('Validation_Dataset.csv')
	X = df.iloc[:, 2:].to_numpy()
	y = np.array([ 0 if x == 'B' else 1 for x in df.iloc[:, 1].tolist()])
	X_valid = df_valid.iloc[:, 2:].to_numpy()
	y_valid = np.array([ 0 if x == 'B' else 1 for x in df_valid.iloc[:, 1].tolist()])
	tester.train(X, y, X_valid, y_valid)