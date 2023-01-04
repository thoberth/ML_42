import numpy as np

# SAME EXERCICE THAN ML00/EX04

def loss_(y, y_hat):
	"""
	Computes the mean squared error of two non-empty numpy.array, without any for loop. The two arrays must have the same dimensions.
	Args:
	y: has to be an numpy.array, a vector.
	y_hat: has to be an numpy.array, a vector.
	Return:
	The half mean squared error of the two vectors as a float.
	None if y or y_hat are empty numpy.array.
	None if y and y_hat does not share the same dimensions.
	None if y or y_hat is not of expected type.
	Raises:
	This function should not raise any Exception.
	"""
	if not (isinstance(y, np.ndarray) or isinstance(y_hat, np.ndarray)) or y.shape != y_hat.shape:
		print("Error y and y_hat must be np.ndarray with same dimensions")
		return None
	squared_error = (y_hat - y) ** 2.
	sum_squared_error = np.sum(squared_error)
	hmse = sum_squared_error / (y.size * 2.)
	return hmse

if __name__ == "__main__":
	X = np.array([0, 15, -9, 7, 12, 3, -21]).reshape((-1, 1))
	Y = np.array([2, 14, -13, 5, 12, 4, -19]).reshape((-1, 1))
	print(loss_(X, Y))
	print(loss_(X, X))
