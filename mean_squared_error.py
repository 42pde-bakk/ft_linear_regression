import argparse
import sys
from math import pow

from linear_regression import LinearRegression


def calculate_error(args: argparse.Namespace):
	linreg = LinearRegression()
	linreg.load_data(args.data_file)
	linreg.load_thetas(args.thetas_file)

	sum_total = sum_of_errors_squared = 0
	for row in linreg.data:
		features, price = row[:-1], row[-1]
		estimated_price = linreg.predict(features)
		sum_of_errors_squared += pow(price - estimated_price, 2)
		sum_total += pow(price, 2)
	mse = sum_of_errors_squared / linreg.data.shape[0]
	sum_total /= linreg.data.shape[0]
	if args.verbose:
		print(f'Total mean squared: {sum_total}, mean squared error: {mse}')
	pct = 100 - mse / sum_total * 100
	print(f'Gives an accuracy of {pct:.2f}% (when within the limits of the training dataset, mind you).')


def parse_arguments():
	parser = argparse.ArgumentParser()
	parser.add_argument('data_file', nargs = '?', action = 'store', help = 'Data.csv', default = 'data.csv')
	parser.add_argument('thetas_file', nargs = '?', action = 'store', help = 'File for the theta values', default = 'thetas.csv')
	parser.add_argument('--verbose', '-v', action = 'store_true', help = 'Show the mean squared error')

	a = parser.parse_args()
	if not a.thetas_file or not a.data_file:
		if not a.thetas_file:
			print(f'Please supply a valid thetas file', file = sys.stderr)
		else:
			print(f'Please supply a valid data.csv file', file = sys.stderr)
		exit(1)
	return a


if __name__ == '__main__':
	argys = parse_arguments()
	calculate_error(argys)
