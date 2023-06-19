import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

if __name__=='__main__':
	header = []
	with open('header.txt', 'r') as f:
		header = f.readlines()
	for i in range(len(header)):
		header[i] = header[i].split(':')[0]
	df = pd.read_csv('data.csv', names=header)
	print(df)