import argparse
import sys
import time

from linear_regression import LinearRegression


def prompt_bonus(linreg: LinearRegression) -> tuple[int, int]:
	params = list(map(int, input(f'Please type the input parameters ({", ".join(linreg.columns[:-1])}): ').split()))
	if len(params) != len(linreg.columns) - 1:
		print(f'Error. Expecting {len(linreg.columns) - 1} values, got {len(params)}.')
		raise KeyError
	if any(param < 0 for param in params):
		raise ValueError
	price = linreg.predict(params)
	return params[0], int(price)


def prompt(linreg: LinearRegression) -> tuple[int, int]:
	mileage = int(input('Please type the mileage in kilometers: '))
	if mileage < 0:
		raise ValueError
	price = linreg.predict(mileage)
	return mileage, int(price)


def predict_car_price(args: argparse.Namespace) -> None:
	linreg = LinearRegression()
	inputs, outputs = [], []
	try:
		linreg.load_data(args.data_file)
		linreg.load_thetas(args.thetas_file)
	except AssertionError:
		print(f'Supplied thetas file ({args.thetas_file}) is invalid.', file = sys.stderr)
		return
	bonus = bool(linreg.thetas.shape[0] > 2)
	while True:
		try:
			if bonus:
				a, b = prompt_bonus(linreg)
			else:
				a, b = prompt(linreg)
			print(f'I predict €{b}.')
			inputs.append(a)
			outputs.append(b)
		except ValueError:
			print('Please input valid parameters.', file = sys.stderr, flush = True)
			time.sleep(0.5)  # Otherwise stderr doesn't flush, ikr
		except KeyError:
			time.sleep(0.5)
		except KeyboardInterrupt:
			if args.verbose:
				linreg.plot_predictions(inputs, outputs)
			break


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
	arguments = parse_arguments()
	predict_car_price(arguments)
