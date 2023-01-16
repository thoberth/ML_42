import numpy as np
import math


def logistic_predict_(x, theta):
	"""
	Computes the vector of prediction y_hat from two non-empty numpy.ndarray. Args:
	x: has to be an numpy.ndarray, a matrice of dimension m * n.
	theta: has to be an numpy.ndarray, a vector of dimension (n + 1) * 1.
	Returns:
	y_hat as a numpy.ndarray, a vector of dimension m * 1.
	None if x or theta are empty numpy.ndarray.
	None if x or theta dimensions are not appropriate.
	Raises:
	This function should not raise any Exception.
	"""
	if not (isinstance(x, np.ndarray) and isinstance(theta, np.ndarray)) or\
			(x.shape not in [(x.shape[0],), (x.shape[0], theta.shape[0] - 1)]) or\
			(theta.shape not in [(theta.shape[0],), (theta.shape[0], 1)]):
		print('logistic predict function error in parameters')
		return None
	x = np.concatenate((np.ones((x.shape[0], 1)), x), axis = 1)
	y_hat = x @ theta
	print(y_hat)
	return y_hat

if __name__ == "__main__":
	x = np.array([4]).reshape((-1, 1))
	print(x)
	theta = np.array([[2], [0.5]])
	print(logistic_predict_(x, theta))

	x2 = np.array([[4], [7.16], [3.2], [9.37], [0.56]])
	print(x2)
	theta2 = np.array([[2], [0.5]])
	# logistic_predict_(x2, theta2)

	x3 = np.array([[0, 2, 3, 4], [2, 4, 5, 5], [1, 3, 2, 7]])
	print(x3)
	theta3 = np.array([[-2.4], [-1.5], [0.3], [-1.4], [0.7]])
	# logistic_predict_(x3, theta3)
