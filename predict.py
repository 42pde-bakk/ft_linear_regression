import argparse
import sys
import time
import numpy as np

from linear_regression import LinearRegression
from parse_args import parse_arguments


def prompt_bonus(linreg: LinearRegression) -> tuple[np.ndarray, int]:
	params = list(map(int, input(f'Please type the input parameters ({", ".join(linreg.columns[:-1])}): ').split()))
	if len(params) != len(linreg.columns) - 1:
		print(f'Error. Expecting {len(linreg.columns) - 1} values, got {len(params)}.')
		raise KeyError
	if any(param < 0 for param in params):
		raise ValueError
	price = linreg.predict(params)
	params.append(price)
	return np.array(params, dtype = float), int(price)


def prompt(linreg: LinearRegression) -> tuple[np.ndarray, int]:
	mileage = int(input('Please type the mileage in kilometers: '))
	if mileage < 0:
		raise ValueError
	price = linreg.predict(mileage)
	return np.array([mileage, price], dtype = float), int(price)


def predict_car_price(args: argparse.Namespace) -> None:
	linreg = LinearRegression()
	preds = []
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
				arr, price = prompt_bonus(linreg)
			else:
				arr, price = prompt(linreg)
			print(f'I predict €{price}.')
			preds.append(arr)
		except ValueError:
			print('Please input valid parameters.', file = sys.stderr, flush = True)
			time.sleep(0.5)  # Otherwise stderr doesn't flush, ikr
		except KeyError:
			time.sleep(0.5)
		except KeyboardInterrupt:
			if args.verbose:
				linreg.plot_predictions(preds = np.array(preds))
			break


if __name__ == '__main__':
	arguments = parse_arguments()
	predict_car_price(arguments)
