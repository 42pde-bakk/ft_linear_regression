import argparse
import sys


def parse_arguments():
	parser = argparse.ArgumentParser('Train logistic regression algorithm')
	parser.add_argument('data_file', nargs = '?', help = 'Filepath for the data.csv file', default = 'data.csv')
	parser.add_argument('thetas_file', nargs = '?', action = 'store', help = 'File for the theta values', default = 'thetas.csv')
	parser.add_argument('--verbose', '-v', action = 'store_true', help = 'Plot graph')
	parser.add_argument('--cprofile', action = 'store_true', help = 'Run cProfile to see where most time is spent.')
	parser.add_argument('--learningrate', '-l', default = 0.1, type = float)
	parser.add_argument('--iterations', '-i', default = 300, type = int)
	arguments = parser.parse_args()
	if not arguments.data_file:
		print(f'Usage: python3 train.py [data.csv]', file = sys.stderr)
		exit(1)
	return arguments
