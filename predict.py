import argparse
import sys
import time

from linear_regression import LinearRegression


def predict_car_price(args: argparse.Namespace):
	linreg = LinearRegression()
	inputs, outputs = [], []
	try:
		linreg.load_data(args.data_file)
		linreg.load_thetas(args.thetas_file)
	except FileNotFoundError:
		print(f'Please supply a valid filename', file = sys.stderr)
		return
	except AssertionError:
		print(f'Supplied file is invalid', file = sys.stderr)
		return
	while True:
		try:
			mileage = int(input('Please type the mileage in kilometers: '))
			if mileage < 0:
				raise ValueError
			price = linreg.predict(mileage)
			inputs.append(mileage)
			outputs.append(int(price))
			print(f'I predict that a car with {mileage} km mileage will have a price of €{int(price)}')
		except ValueError:
			print('Please input a valid mileage in kilometers (please give an int).', file = sys.stderr, flush = True)
			time.sleep(0.5)  # Otherwise stderr doesn't flush, ikr
		except KeyboardInterrupt:
			if args.verbose:
				linreg.plot_predictions(inputs, outputs)
			return


def parse_arguments():
	parser = argparse.ArgumentParser()
	parser.add_argument('data_file', nargs = '?', action = 'store', help = 'Data.csv', default = 'data.csv')
	parser.add_argument('thetas_file', nargs = '?', action = 'store', help = 'File for the theta values',
	                    default = 'thetas.csv')
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
	arguments = parse_arguments()
	predict_car_price(arguments)
