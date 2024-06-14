import argparse
from typing import List
import sys
import os
from loss_functions import binarycrossentropy


def control_acti(activation):
	activations_available = ['ReLu', 'sigmoid', 'tanh', 'softmax']
	if activation not in activations_available:
		raise argparse.ArgumentTypeError(f"{activation} is not an available activation function\n\tActivation function: {activations_available}")
	return activation

def control_layers(layer_size):
	try:
		layer_size = int(layer_size)
	except ValueError:
		if not 1 <= layer_size <= 100:
			raise argparse.ArgumentTypeError(f"{layer_size} is not a correct layer lenght", file=sys.stderr)
	return layer_size

def control_epochs(epochs):
	try:
		value = int(epochs)
		if value >= 1:
			return value
		else:
			raise argparse.ArgumentTypeError(f"{epochs} is not superior or equl to 1")
	except ValueError:
		raise argparse.ArgumentTypeError(f"{epochs} is not a valid int")


def control_batchsize(size):
	try:
		value = int(size)
		if value >= 1:
			return value
		else:
			raise argparse.ArgumentTypeError(f"{size} is not superior or equl to 1")
	except ValueError:
		raise argparse.ArgumentTypeError(f"{size} is not a valid int")


def control_lr(size):
	try:
		value = float(size)
		if 1e-8 <= value <= 0.1:
			return value
		else:
			raise argparse.ArgumentTypeError(f"{size} is not between 1e-8 and 0.1")
	except ValueError:
		raise argparse.ArgumentTypeError(f"{size} is not a valid float")


def control_file(file_path):
	if not os.path.exists(file_path):
		raise argparse.ArgumentTypeError(f"The file '{file_path}' do not exists")
	if not os.access(file_path, os.R_OK):
		raise argparse.ArgumentTypeError(f"Cannot open the file '{file_path}' in read mode")
	return file_path


def control_lossfunction(loss_function):
	func = None
	if loss_function == 'binarycrossentropy':
		return loss_function
	else:
		raise argparse.ArgumentError(f"{loss_function} is not a valid loss function\n\tAvailable loss function : 'binarycrossentropy'")

def validate_metrics(metrics):
	valid_metrics = {'f1 score', 'recall', 'auc'}
	metrics = metrics.lower().replace('_', ' ')
	if not metrics in valid_metrics:
		raise argparse.ArgumentTypeError(f"{metrics} is not a valid metrics\n\tAvailable metrics are {valid_metrics}")
	return metrics

def validate_momentum(momentum):
	try:
		value = float(momentum)
		if  0 < value < 1:
			return value
		else:
			raise argparse.ArgumentTypeError(f"{momentum} is not between 0 and 1")
	except ValueError:
		raise argparse.ArgumentTypeError(f"{momentum} is not a float")