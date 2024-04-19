import pandas as pd
import argparse
from utils import control_batchsize, control_epochs, control_lr, control_file, control_layers, control_lossfunction
from perceptron import Perceptron
from activation_functions import softmax, sigmoid
from loss_functions import binarycrossentropy
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="This create a NN Perceptron\
									and then train with chosen args")
	parser.add_argument('-hl', '--layer', nargs='+', default=[8, 8], type=int, action='store', help='Size of number of layers')

	parser.add_argument('-e', '--epochs', type=control_epochs, default=100, action='store', help='Number of epochs while training')

	parser.add_argument('-l', '--loss', type=control_lossfunction, default=binarycrossentropy, choices=['binarycrossentropy'], action='store', help='The loss Function')

	parser.add_argument('-b', '--batch_size', type=control_batchsize, action='store')

	parser.add_argument('-lr', '--learning_rate', type=control_lr, action='store', help='The value of the learning rate')

	parser.add_argument('--csv', default='Training_Dataset.csv', type=control_file, action='store', help='The Dataset to train')

	parser.add_argument('--csv2', default='Validation_Dataset.csv', type=control_file, action='store', help='The Dataset to evaluate')

	args = parser.parse_args()
	control_layers(args.layer)

	df = pd.read_csv(args.csv)
	df_valid = pd.read_csv(args.csv2)
	y = df.iloc[:, 1].apply(lambda x: 0 if x == 'B' else 1).to_numpy()
	X = df.iloc[:, 2:].to_numpy()
	# sns.pairplot(df.iloc[:, 1:10], hue='1')
	# plt.show()
	X_valid = df_valid.iloc[:, 2:].to_numpy()
	y_valid = df_valid.iloc[:, 1].apply(lambda x: 0 if x == 'B' else 1).to_numpy()
	p = Perceptron(layers=args.layer, epochs=args.epochs, loss_function=args.loss, lr=args.learning_rate).train(X.T, y, X_valid, y_valid)
