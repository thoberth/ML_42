import numpy as np
import math

def sigmoid_(x):
	"""
	Compute the sigmoid of a vector.
	Args:
	x: has to be a numpy.ndarray of shape (m, 1).
	Returns:
	The sigmoid value as a numpy.ndarray of shape (m, 1).
	None if x is an empty numpy.ndarray.
	Raises:
	This function should not raise any Exception.
	"""
	if not isinstance(x, np.ndarray) or x.shape not in [(x.shape[0], ), (x.shape[0], 1)]:
		print('Sigmoid function error: x is not an array or shape is wrong')
		return None
	x = x.astype('float64')
	for i in range(x.shape[0]):
		x[i][0] = (1. / (1. + math.exp(-x[i][0])))
	return x

if __name__=="__main__":
	x = np.array([[-4]])
	print(sigmoid_(x))

	x = np.array([[2]])
	print(sigmoid_(x))

	x = np.array([[-4], [2], [0]])
	print(sigmoid_(x))
