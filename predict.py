import sys
import argparse
from linear_regression import LinearRegression


def predict_car_price(args: argparse.Namespace):
	linreg = LinearRegression()
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
			price = linreg.predict(mileage)
			print(f'I predict that a car with {mileage} km mileage will have a price of €{int(price)}')
		except ValueError:
			print('Please input a valid mileage in kilometers (please give an int).', file = sys.stderr)
		except KeyboardInterrupt:
			return


def parse_arguments():
	parser = argparse.ArgumentParser()
	parser.add_argument('thetas_file', nargs = '?',  action = 'store', help = 'File for the theta values', default = None)
	parser.add_argument('data_file', nargs = '?', action = 'store', help = 'Data.csv', default = None)
	parser.add_argument('--verbose', '-v', action = 'store_true', help='Show the mean squared error')

	a = parser.parse_args()
	if not a.thetas_file or not a.data_file:
		if not a.thetas_file:
			print(f'Please supply a valid thetas file', file = sys.stderr)
		else:
			print(f'Please supply a valid data.csv file', file = sys.stderr)
		exit(1)
	return a

if __name__ == '__main__':
	if len(sys.argv) < 3:
		print(f'Usage: python3 predict.py [thetas.csv] [data.csv]', file = sys.stderr)
	else:
		predict_car_price(*sys.argv[1:])
