from utils import *

if __name__=='__main__':
	header = []
	with open('header.txt', 'r') as f:
		header = f.readlines()
	for i in range(len(header)):
		header[i] = header[i].split(':')[0]
	df = pd.read_csv('data.csv', names=header)
	df.drop(['ID'], axis=1, inplace=True)
	X = df.drop(['Diagnosis'], axis=1).to_numpy()
	y = df['Diagnosis'].to_numpy().reshape(-1, 1)
	arg = ['layer', 'epochs', 'loss']