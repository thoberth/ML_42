import math
import numpy as np


def loss_(y, y_hat):
	"""Computes the half mean squared error of two non-empty numpy.array, without any for loop. The two arrays must have the same dimensions.
	Args:
	  y: has to be an numpy.array, a vector.
	  y_hat: has to be an numpy.array, a vector.
	Returns:
	  The half mean squared error of the two vectors as a float.
	  None if y or y_hat are empty numpy.array.
	  None if y and y_hat does not share the same dimensions.
	Raises:
	  This function should not raise any Exceptions.
	"""
	if not (isinstance(y, np.ndarray) and isinstance(y_hat, np.ndarray)) or\
		y.shape not in [(y.size,), (y.size, 1)] or y_hat.shape not in [(y_hat.size,), (y_hat.size, 1)]:
		print("Error arguments are wrong")
		return None
	squared_error = (y_hat - y) ** 2
	print(squared_error)
	sum_squared_error = sum(squared_error)
	print(sum_squared_error)
	mse = sum_squared_error / (y.size*2)
	return mse

if __name__=="__main__":
	X = np.array([[0], [15], [-9], [7], [12], [3], [-21]])
	Y = np.array([[2], [14], [-13], [5], [12], [4], [-19]])

	print(loss_(X, Y))
	print(loss_(X, X))
