import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def normalize(data: np.ndarray, minmax: bool):
	result = data.copy()
	if minmax:
		return (result - result.min()) / (result.max() - result.min())
	return (result - result.mean()) / result.std()


def plot_data(x, y, xlabel, ylabel):
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.plot(x[:, 0], y, 'bo')
	plt.show()


class LinearRegression:
	def __init__(self, learning_rate = 0.1, iterations = 300):
		self.learning_rate = learning_rate
		self.iterations = iterations
		self.thetas = np.ndarray
		self.data = np.ndarray
		self.x, self.y = np.ndarray, np.ndarray

	@staticmethod
	def __estimate_price(mileage: int, thetas: np.ndarray) -> float:
		return thetas[0] + thetas[1] * float(mileage)

	def __update_thetas(self):
		tmp_thetas = np.zeros(shape = self.thetas.shape, dtype = float)
		m = self.x.shape[0]
		for i in range(0, m):
			mileage, price = self.x[i], self.y[i]
			toc = LinearRegression.__estimate_price(mileage, self.thetas) - price
			tmp_thetas[0] += toc
			tmp_thetas[1] += toc * mileage
		for theta in range(0, tmp_thetas.shape[0]):
			tmp_thetas[theta] *= self.learning_rate * (1.0 / m)
			self.thetas[theta] -= tmp_thetas[theta]

	def load_data(self, filename: str, minmax_normalizing: bool = True):
		np.set_printoptions(suppress = True)
		try:
			df = pd.read_csv(filename, sep = ',', index_col = False)
		except FileNotFoundError:
			print('Please supply a valid path to the data.csv file.', file = sys.stderr)
			exit(1)
		self.data = np.array(df, dtype = float)
		self.x = normalize(self.data[:, :-1], minmax_normalizing)
		self.y = normalize(self.data[:, -1], minmax_normalizing)
		return self.data[:, :-1], self.data[:, -1]

	def save_thetas(self) -> None:
		with open('thetas.csv', 'w') as f:
			for th in self.thetas:
				f.write(str(float(th)) + '\n')

	def load_thetas(self, filename: str):
		try:
			with open(filename, 'r') as f:
				lines = [float(row) for row in f.read().splitlines()]
		except FileNotFoundError:
			print('Please supply a valid path to the thetas file.', file = sys.stderr)
			exit(1)
		self.thetas = np.array(lines, dtype = float)
		assert len(self.thetas) == 2

	def train(self):
		self.thetas = np.zeros(shape = (self.x.shape[1] + 1, 1), dtype = float)
		for iteration in range(self.iterations):
			self.__update_thetas()

	def predict(self, mileage) -> float:
		kms, prices = self.data[:, 0], self.data[:, -1]
		min_km, max_km = min(kms), max(kms)
		min_price, max_price = min(prices), max(prices)

		normalized_mileage = (mileage - min_km) / (max_km - min_km)
		normalized_price = self.__estimate_price(normalized_mileage, self.thetas)
		if normalized_price < 0:
			# print(f'Warning. Given mileage ({mileage}) too high', file = sys.stderr)
			return 0
		return normalized_price * (max_price - min_price) + min_price

	def get_regression_line(self):
		min_x, max_x = min(self.data[:, 0]), max(self.data[:, 0])
		min_y, max_y = min(self.data[:, -1]), max(self.data[:, -1])
		line_x, line_y = [min_x, max_x], []
		for point in line_x:
			normalized_x = (point - min_x) / (max_x - min_x)
			point = self.thetas[1] * normalized_x + self.thetas[0]
			if point != 0:
				denormalized_y = point * (max_y - min_y) + min_y
			else:
				denormalized_y = 0
			line_y.append(denormalized_y)
		return line_x, line_y

	def plot(self):
		plt.xlabel('Mileage')
		plt.ylabel('Price')
		plt.plot(self.data[:, 0], self.data[:, -1], 'bo')
		line_x, line_y = self.get_regression_line()
		# print(f'line_x={line_x}, line_y={line_y}')
		plt.plot(line_x, line_y, 'tab:olive', label="Best line")
		# plt.plot(self.x, b0 + b1 * self.x, c = 'r', linewidth = 5, alpha=.5, solid_capstyle='round')
		plt.show()
