import sys
import os
import argparse
import cProfile
import pstats
from math import pow
from linear_regression import LinearRegression


def calculate_error(args: argparse.Namespace):
	linreg = LinearRegression()
	linreg.load_data(args.data)
	linreg.load_thetas(args.thetas)

	sum_of_errors_squared = 0
	sum_total = 0
	for item in linreg.data:
		mileage, actual_price = item
		estimated_price = linreg.predict(mileage)
		sum_of_errors_squared += pow(actual_price - estimated_price, 2)
		sum_total += pow(actual_price, 2)
	mse = sum_of_errors_squared / linreg.data.shape[0]
	sum_total /= linreg.data.shape[0]
	print(f'Total mean squared: {sum_total}, mean squared error: {mse}')
	pct = 100 - mse / sum_total * 100
	print(f'Gives an accuracy of {pct:.2f}% (when within the limits of the training dataset, mind you).')


def parse_arguments():
	parser = argparse.ArgumentParser()
	parser.add_argument('--thetas', '-t', action = 'store', help = 'File for the theta values',default = None)
	parser.add_argument('--data', '-d', action = 'store', help = 'Data.csv', default = None)

	a = parser.parse_args()
	if not a.thetas or not a.data:
		exit(1)
	return a


if __name__ == '__main__':
	argys = parse_arguments()
	calculate_error(argys)
