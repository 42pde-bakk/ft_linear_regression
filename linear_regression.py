import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def normalize(data: np.ndarray, minmax: bool):
	result = data.copy()
	if minmax:
		return (result - result.min()) / (result.max() - result.min())
	return (result - result.mean()) / result.std()


class LinearRegression:
	def __init__(self, learning_rate = 0.1, iterations = 300):
		self.learning_rate = learning_rate
		self.iterations = iterations
		self.thetas = np.ndarray
		self.data = np.ndarray
		self.columns = list()
		self.bonus = False
		self.x, self.y = np.ndarray, np.ndarray

	@staticmethod
	def __estimate_price_bonus(arr_x: np.ndarray, thetas: np.ndarray) -> float:
		return thetas[0] + sum([th * float(x) for th, x in zip(thetas[1:], arr_x)])

	def __update_thetas_bonus(self):
		tmp_thetas = np.zeros(shape = self.thetas.shape, dtype = float)
		m, n = self.x.shape
		for i in range(0, m):
			arr_x, price = self.x[i], self.y[i]
			toc = LinearRegression.__estimate_price_bonus(arr_x, self.thetas) - price
			tmp_thetas[0] += toc
			for i2 in range(1, arr_x.shape[0] + 1):
				tmp_thetas[i2] += toc * float(arr_x[i2 - 1])
		for theta in range(0, tmp_thetas.shape[0]):
			tmp_thetas[theta] *= self.learning_rate * (1.0 / m)
			self.thetas[theta] -= tmp_thetas[theta]

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
			self.data = np.array(df, dtype = float)
			self.columns = [str(col) for col in df.columns]
			self.x = normalize(self.data[:, :-1], minmax_normalizing)
			self.y = self.data[:, -1]
			self.bonus = bool(self.x.shape[1] >= 2)
			return self.data[:, :-1], self.data[:, -1]
		except FileNotFoundError:
			print(f'Error finding {filename}.', file = sys.stderr)
			exit(1)
		except ValueError:
			print(f'Are you sure {filename} contains the right data?', file = sys.stderr)
			exit(1)

	def save_thetas(self, thetas_file: str) -> None:
		with open(thetas_file, 'w') as f:
			for th in self.thetas:
				f.write(str(float(th)) + '\n')

	def load_thetas(self, filename: str):
		try:
			with open(filename, 'r') as f:
				thetas = [float(row) for row in f.read().splitlines()]
		except FileNotFoundError:
			print('Please supply a valid path to the thetas file.', file = sys.stderr)
			exit(1)
		self.thetas = np.array(thetas, dtype = float)
		assert self.thetas.shape[0] == self.x.shape[1] + 1

	def train(self):
		self.thetas = np.zeros(shape = (self.x.shape[1] + 1, 1), dtype = float)
		for iteration in range(self.iterations):
			if self.bonus:
				self.__update_thetas_bonus()
			else:
				self.__update_thetas()

	def __predict_bonus(self, params: list[int]) -> float:
		mins = [min(self.data[:, i]) for i in range(self.data.shape[1])]
		maxs = [max(self.data[:, i]) for i in range(self.data.shape[1])]
		normalized = [(param - mins[i]) / (maxs[i] - mins[i]) for i, param in enumerate(params)]
		return max(0.0, self.__estimate_price_bonus(np.array(normalized, dtype = float), self.thetas))

	def predict(self, mileage) -> float:
		if self.bonus:
			return self.__predict_bonus(mileage)
		kms, prices = self.data[:, 0], self.data[:, -1]
		min_km, max_km = min(kms), max(kms)
		normalized_mileage = (mileage - min_km) / (max_km - min_km)
		return max(0.0, self.__estimate_price(normalized_mileage, self.thetas))

	def __get_regression_line(self, col_nb: int = 0):
		min_x, max_x = min(self.data[:, col_nb]), max(self.data[:, col_nb])
		xy1 = min_x, float(self.predict(min_x))
		xy2 = max_x, float(self.predict(max_x))
		return xy1, xy2

	def plot_bonus(self, preds: np.ndarray = None):
		plt.close('all')
		figure, axes = plt.subplots(self.data.shape[1] - 1)
		for col_nb in range(0, self.data.shape[1] - 1):
			axes[col_nb].set_xlabel(self.columns[col_nb])
			axes[col_nb].set_ylabel(self.columns[-1])
			axes[col_nb].plot(self.data[:, col_nb], self.data[:, -1], 'ok')
			if preds is not None:
				axes[col_nb].plot(preds[:, col_nb], preds[:, -1], 'bo')
		plt.show()

	def plot(self):
		if self.bonus:
			return self.plot_bonus()
		plt.xlabel(self.columns[0])
		plt.ylabel(self.columns[1])
		plt.plot(self.data[:, 0], self.data[:, -1], 'ok')
		xy1, xy2 = self.__get_regression_line()
		plt.axline(xy1 = xy1, xy2 = xy2, color = 'purple', label = 'Regression line')
		plt.show()

	def plot_predictions(self, preds: np.ndarray):
		plt.close('all')
		if self.bonus:
			return self.plot_bonus(preds = preds)
		plt.plot(preds[:, 0], preds[:, -1], 'bo')
		self.plot()
