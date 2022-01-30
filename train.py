import sys
import os
from linear_regression import LinearRegression


def train(filename):
    if not filename or not os.path.exists(filename):
        print(f'Please supply a valid data.csv file', file = sys.stderr)
        return
    linreg = LinearRegression(learning_rate = 0.1, iterations = 300)
    linreg.load_data('data.csv')

    linreg.train()
    linreg.save_thetas()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(f'Usage: python3 train.py [data.csv]', file = sys.stderr)
    else:
        train(sys.argv[1])
