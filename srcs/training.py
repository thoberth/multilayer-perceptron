import pandas as pd
import argparse
from utils import control_batchsize, control_epochs, control_lr, control_file, control_layers, control_lossfunction, control_acti, validate_metrics
from perceptron import Perceptron
from activation_functions import softmax, sigmoid
from loss_functions import binarycrossentropy
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="This create a NN Perceptron\
									and then train with chosen args")
	parser.add_argument('-hl', '--layer', nargs='+', default=[8, 8], type=control_layers, help='Size of number of layers')

	parser.add_argument('-a', '--acti', nargs='+', type=control_acti, default=['sigmoid', 'sigmoid'], help='Activation function of the corresponding layer')

	parser.add_argument('-e', '--epochs', type=control_epochs, default=100, action='store', help='Number of epochs while training')

	parser.add_argument('-l', '--loss', type=control_lossfunction, default=binarycrossentropy, choices=['binarycrossentropy'], action='store', help='The loss Function')

	parser.add_argument('-b', '--batch_size', type=control_batchsize, action='store')

	parser.add_argument('-lr', '--learning_rate', type=control_lr, action='store', help='The value of the learning rate')

	parser.add_argument('--csv', default='Training_Dataset.csv', type=control_file, action='store', help='The Dataset to train')

	parser.add_argument('--csv2', default='Validation_Dataset.csv', type=control_file, action='store', help='The Dataset to evaluate')

	parser.add_argument('--early_stop', action='store_true', help='Early stop to stop training before the number of epochs if needed')

	parser.add_argument('-s', '--show', action='store_true', help='Show loss and accuracy_curves')

	parser.add_argument('--metrics', nargs='+', default=[], type=validate_metrics, help="Specify one or more metrics. Valid options are 'f1_Score', 'accuracy', 'recall'")

	args = parser.parse_args()

	df = pd.read_csv(args.csv)
	df_valid = pd.read_csv(args.csv2)
	y = np.array([ 0 if x == 'B' else 1 for x in df.iloc[:, 1].tolist()])
	X = df.iloc[:, 2:].to_numpy()
	X_valid = df_valid.iloc[:, 2:].to_numpy()
	y_valid = np.array([ 0 if x == 'B' else 1 for x in df_valid.iloc[:, 1].tolist()])
	p = Perceptron(layers=args.layer, activation=args.acti, epochs=args.epochs, loss_function=args.loss,\
				lr=args.learning_rate, early_stop=args.early_stop, metrics=args.metrics, show_metrics=args.show)
	p.train(X, y, X_valid, y_valid)
	p.save_model(f'Training_{datetime.now().strftime("%m_%d_%H:%M:%S")}')