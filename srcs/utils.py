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
			raise argparse.ArgumentTypeError(f"{layer_size} n'est pas une correcte taille de layers", file=sys.stderr)
	return layer_size

def control_epochs(epochs):
	try:
		value = int(epochs)
		if value >= 1:
			return value
		else:
			raise argparse.ArgumentTypeError(f"{epochs} n'est pas superieur ou egale a 1")
	except ValueError:
		raise argparse.ArgumentTypeError(f"{epochs} n'est pas un int valide")


def control_batchsize(size):
	try:
		value = int(size)
		if value >= 1:
			return value
		else:
			raise argparse.ArgumentTypeError(f"{size} n'est pas superieur ou egale a 1")
	except ValueError:
		raise argparse.ArgumentTypeError(f"{size} n'est pas un int valide")


def control_lr(size):
	try:
		value = float(size)
		if 1e-8 <= value <= 0.1:
			return value
		else:
			raise argparse.ArgumentTypeError(f"{size} n'est pas compris entre 1e-8 et 0.1")
	except ValueError:
		raise argparse.ArgumentTypeError(f"{size} n'est pas un nombre flottant valide")


def control_file(file_path):
	if not os.path.exists(file_path):
		raise argparse.ArgumentTypeError(f"Le fichier '{file_path}' n'existe pas")
	if not os.access(file_path, os.R_OK):
		raise argparse.ArgumentTypeError(f"Impossible d'ouvrir le fichier '{file_path}' en lecture")
	return file_path


def control_lossfunction(loss_function):
	func = None
	if loss_function == 'binarycrossentropy':
		func = binarycrossentropy
	else:
		raise argparse.ArgumentError(f"{loss_function} n'est pas une fonction cout valide")

def validate_metrics(metrics):
	valid_metrics = {'f1 score', 'recall', 'auc'}
	metrics = metrics.lower().replace('_', ' ')
	if not metrics in valid_metrics:
		raise argparse.ArgumentTypeError(f"{metrics} est un nom de metrique invalide!")
	return metrics