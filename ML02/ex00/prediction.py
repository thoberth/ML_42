import numpy as np


def predict_(x, theta):
	"""
	Computes the prediction vector y_hat from two non-empty numpy.array. Args:
	x: has to be an numpy.array, a vector of dimensions m * n.
	theta: has to be an numpy.array, a vector of dimensions (n + 1) * 1.
	Return:
	y_hat as a numpy.array, a vector of dimensions m * 1.
	None if x or theta are empty numpy.array.
	None if x or theta dimensions are not appropriate.
	None if x or theta is not of expected type.
	Raises:
	This function should not raise any Exception.
	"""
	x = np.concatenate((np.ones((x.shape[0], 1)), x), axis = 1)
	y_hat = x @ theta
	return y_hat

if __name__ == "__main__":
	x = np.arange(1, 13).reshape((4, -1))
	theta1 = np.array([5, 0, 0, 0]).reshape((-1, 1))
	print(predict_(x, theta1), end='\n\n')

	theta2 = np.array([0, 1, 0, 0]).reshape((-1, 1))
	print(predict_(x, theta2), end='\n\n')

	theta3 = np.array([-1.5, 0.6, 2.3, 1.98]).reshape((-1, 1))
	print(predict_(x, theta3), end='\n\n')

	theta4 = np.array([-3, 1, 2, 3.5]).reshape((-1, 1))
	print(predict_(x, theta4), end='\n\n')
