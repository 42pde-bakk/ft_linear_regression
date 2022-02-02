import argparse
import cProfile
import pstats
import sys

from linear_regression import LinearRegression


def train(arguments: argparse.Namespace):
	if arguments.cprofile:
		pr = cProfile.Profile()
		pr.enable()
	linreg = LinearRegression(learning_rate = 0.1, iterations = 300)
	linreg.load_data(arguments.data_file, minmax_normalizing = True)

	linreg.train()
	linreg.save_thetas(arguments.thetas_file)
	print('Finished generating thetas, saved them to thetas.csv')
	if arguments.cprofile:
		pr.disable()
		stats = pstats.Stats(pr)
		stats.sort_stats('tottime').print_stats(10)
	if arguments.verbose:
		linreg.plot()


def parse_arguments():
	parser = argparse.ArgumentParser('Train logistic regression algorithm')
	parser.add_argument('data_file', nargs = '?', help = 'Filepath for the data.csv file', default = 'data.csv')
	parser.add_argument('thetas_file', nargs = '?', action = 'store', help = 'File for the theta values',
	                    default = 'thetas.csv')
	parser.add_argument('--verbose', '-v', action = 'store_true', help = 'Plot graph')
	parser.add_argument('--cprofile', action = 'store_true', help = 'Run cProfile to see where most time is spent.')
	arguments = parser.parse_args()
	if not arguments.data_file:
		print(f'Usage: python3 train.py [data.csv]', file = sys.stderr)
		exit(1)
	return arguments


if __name__ == '__main__':
	args = parse_arguments()
	train(args)
