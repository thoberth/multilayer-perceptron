from utils import *

def zscore_standardization(df: pd.DataFrame):
	'''    Calculate the mean and standard deviation of each feature.
	Subtract the mean from each value and then divide by the standard deviation.
	This method ensures that the transformed data has a mean of 0 and a standard deviation of 1.'''
	print("zscore standardization is processing ... ")
	for feature, values in df.items():
		if feature > 1:
			df[feature] = (values - values.mean()) / values.std()

def minmax_standardization(df: pd.DataFrame):
	print("Min max standardization is processing ... ")
	for feature, values in df.items():
		if feature > 1:
			df[feature] = (values - values.min()) / (values.max() - values.min())

def control_training_size(size):
	try:
		value = float(size)
		if 0.2 <= value <= 1:
			return value
		else:
			raise argparse.ArgumentTypeError(f"{size} n'est pas compris entre 0.2 et 1")
	except ValueError:
		raise argparse.ArgumentTypeError(f"{size} n'est pas un nombre flottant valide")

def control_random_state(seed):
	try:
		value = int(seed)
		if value > 0:
			return value
		else:
			raise argparse.ArgumentTypeError(f"{seed} doit etre superieur a 0")
	except ValueError:
		raise argparse.ArgumentTypeError(f"{seed} n'est pas superieur a 0")

def standardization(method: str, df: pd.DataFrame):
	if method == 'minmax':
		minmax_standardization(df)
	else :
		zscore_standardization(df)

def train_test_split(df: pd.DataFrame, size: float, seed: int)-> List[pd.DataFrame] :
	df = df.sample(frac=1, random_state=seed)

	# Définir le pourcentage pour le premier DataFrame
	percentage_df1 = size

	# Calculer le nombre de lignes pour le premier DataFrame
	size_df1 = int(len(df) * percentage_df1)

	# Diviser le DataFrame en deux
	df1 = df.iloc[:size_df1]
	df2 = df.iloc[size_df1:]

	return [df1, df2]


if __name__=='__main__':
	parser = argparse.ArgumentParser(description="This program first preprocess data and then \
								  split Data into two dataset Training and Validation.")
	parser.add_argument('-r', '--random_state',  action='store', 
					 type=control_random_state, nargs=1, 
					 default=42, help='Seed for shuflling the dataset before split')
	parser.add_argument('-t', '--training_size',  action='store', 
					 type=control_training_size, nargs=1, 
					 default=0.8, help='Size of Training Dataset')
	parser.add_argument('-s', '--standardization', action='store' ,
					 type=str, choices=['minmax', 'zscore'],
					 default='minmax', help='Standardization method for data')
	args = parser.parse_args()

	df = pd.read_csv('data/data.csv', header=None)
	standardization(args.standardization, df)
	test, validation = train_test_split(df, args.training_size, args.random_state)
	test.to_csv('Training_Dataset.csv', index=False)
	validation.to_csv('Validation_Dataset.csv', index=False)
