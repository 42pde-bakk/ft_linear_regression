import argparse
import cProfile
import pstats

from linear_regression import LinearRegression
from parse_args import parse_arguments


def train(arguments: argparse.Namespace):
	if arguments.cprofile:
		pr = cProfile.Profile()
		pr.enable()
	linreg = LinearRegression(learning_rate = args.learningrate, iterations = args.iterations)
	print(f'learningrate = {args.learningrate}, iters={args.iterations}')
	linreg.load_data(arguments.data_file, minmax_normalizing = True)

	linreg.train()
	linreg.save_thetas(arguments.thetas_file)
	print('Finished generating thetas, saved them to thetas.csv')
	if arguments.cprofile:
		pr.disable()
		stats = pstats.Stats(pr)
		stats.sort_stats('tottime').print_stats(10)
	if arguments.verbose:
		linreg.plot()


if __name__ == '__main__':
	args = parse_arguments()
	train(args)
