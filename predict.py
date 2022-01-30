import sys
from linear_regression import LinearRegression


def predict_car_price(filename: str):
	linreg = LinearRegression()
	try:
		linreg.load_data('data/data.csv')
		linreg.load_thetas(filename)
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
			print(f'I predict that a car with {mileage} km mileage will have a price of ${price}')
		except ValueError:
			print('Please input a valid mileage in kilometers (please give an int).', file = sys.stderr)
		except KeyboardInterrupt:
			return


if __name__ == '__main__':
	if len(sys.argv) < 2:
		print(f'Usage: python3 predict.py [thetas.csv] [data.csv]', file = sys.stderr)
	else:
		predict_car_price(sys.argv[1])
