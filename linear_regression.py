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

	def load_data(self, filename: str):
		np.set_printoptions(suppress = True)
		df = pd.read_csv(filename, sep = ',', index_col = False)
		self.data = np.array(df, dtype = float)
		self.x = normalize(self.data[:, :-1], True)
		self.y = normalize(self.data[:, -1], True)
		return self.data[:, :-1], self.data[:, -1]

	def save_thetas(self) -> None:
		with open('thetas.csv', 'w') as f:
			for th in self.thetas:
				f.write(str(float(th)) + '\n')

	def load_thetas(self, filename: str):
		with open(filename, 'r') as f:
			lines = [float(row) for row in f.read().splitlines()]
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
		if normalized_price <= 0:
			print(f'Warning. Given mileage too high', file = sys.stderr)
			return 0
		return normalized_price * (max_price - min_price) + min_price

	def plot(self):
		plt.xlabel('Mileage')
		plt.ylabel('Price')
		plt.plot(self.data[:, 0], self.data[:, -1], 'bo')
		plt.show()
