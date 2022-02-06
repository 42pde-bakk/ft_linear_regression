import argparse
import sys
from math import pow

from linear_regression import LinearRegression
from parse_args import parse_arguments


def calculate_error(args: argparse.Namespace):
	linreg = LinearRegression()
	try:
		linreg.load_data(args.data_file)
		linreg.load_thetas(args.thetas_file)
	except AssertionError:
		print(f'Supplied thetas file ({args.thetas_file}) is invalid.', file = sys.stderr)
		return

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


if __name__ == '__main__':
	argys = parse_arguments()
	calculate_error(argys)
