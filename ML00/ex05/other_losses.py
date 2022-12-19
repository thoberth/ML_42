from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statistics import mean
import numpy as np
import math


def mse_(y, y_hat):
	"""
	Description:
	Calculate the MSE between the predicted output and the real output.
	Args:
	y: has to be a numpy.array, a vector of dimension m * 1.
	y_hat: has to be a numpy.array, a vector of dimension m * 1.
	Returns:
	mse: has to be a float.
	None if there is a matching dimension problem.
	Raises:
	This function should not raise any Exceptions.
	"""
	if not (isinstance(y, np.ndarray) and isinstance(y_hat, np.ndarray)) or\
		y.shape not in [(y.size,), (y.size, 1)] or y_hat.shape not in [(y_hat.size,), (y_hat.size, 1)]:
		print("Error arguments are wrong")
		return None
	squared_error = (y_hat - y) ** 2
	sum_squared_error = sum(squared_error)
	mse = sum_squared_error / (y.size)
	return mse

def rmse_(y, y_hat):
	"""
	Description:
	Calculate the RMSE between the predicted output and the real output.
	Args:
	y: has to be a numpy.array, a vector of dimension m * 1.
	y_hat: has to be a numpy.array, a vector of dimension m * 1.
	Returns:
	rmse: has to be a float.
	None if there is a matching dimension problem.
	Raises:
	This function should not raise any Exceptions.
	"""
	rmse = math.sqrt(mse_(y, y_hat))
	return rmse


def mae_(y, y_hat):
	"""
	Description:
	Calculate the MAE between the predicted output and the real output.
	Args:
	y: has to be a numpy.array, a vector of dimension m * 1.
	y_hat: has to be a numpy.array, a vector of dimension m * 1.
	Returns:
	mae: has to be a float.
	None if there is a matching dimension problem.
	Raises:
	This function should not raise any Exceptions.
	"""
	if not (isinstance(y, np.ndarray) and isinstance(y_hat, np.ndarray)) or\
		y.shape not in [(y.size,), (y.size, 1)] or y_hat.shape not in [(y_hat.size,), (y_hat.size, 1)]:
		print("Error arguments are wrong")
		return None
	absolute_error = abs(y_hat - y)
	sum_absolute_error = sum(absolute_error)
	mae = sum_absolute_error / (y.size)
	return mae


def r2score_(y, y_hat):  # coefficient of determination
	"""
	Description:
	Calculate the R2score between the predicted output and the output.
	Args:
	y: has to be a numpy.array, a vector of dimension m * 1.
	y_hat: has to be a numpy.array, a vector of dimension m * 1.
	Returns:
	r2score: has to be a float.
	None if there is a matching dimension problem.
	Raises:
	This function should not raise any Exceptions.
	"""
	if not (isinstance(y, np.ndarray) and isinstance(y_hat, np.ndarray)) or\
		y.shape not in [(y.size,), (y.size, 1)] or y_hat.shape not in [(y_hat.size,), (y_hat.size, 1)]:
		print("Error arguments are wrong")
		return None
	sum_squared_regression = sum((y - y_hat)**2)
	total_sum_of_squares = sum((y - mean(y))**2)
	r2score_var = 1 - (sum_squared_regression/total_sum_of_squares)
	return r2score_var

if __name__ == "__main__":
	x = np.array([0, 15, -9, 7, 12, 3, -21])
	y = np.array([2, 14, -13, 5, 12, 4, -19])
	print('/// MEAN SQUARED ERROR ///')
	print('My implementation:', end=' ')
	print(mse_(x, y))
	print('Sklearn function:', end=' ')
	print(mean_squared_error(x,y))
	print('/// ROOT MEAN SQUARED ERROR ///')
	print('My implementation:', end=' ')
	print(rmse_(x,y))
	print('Sklearn function:', end=' ')
	print(math.sqrt(mean_squared_error(x,y)))
	print('/// MEAN ABSOLUTE ERROR ///')
	print('My implementation:', end=' ')
	print(mae_(x,y))
	print('Sklearn function:', end=' ')
	print(mean_absolute_error(x,y))
	print('/// R2-SCORE ///')
	print('My implementation:', end=' ')
	print(r2score_(x,y))
	print('Sklearn function:', end=' ')
	print(r2_score(x,y))