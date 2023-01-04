import numpy as np


def minmax(x):
	"""Computes the normalized version of a non-empty numpy.ndarray using the min-max standardization. Args:
	x: has to be an numpy.ndarray, a vector.
	Returns:
	x’ as a numpy.ndarray.
	None if x is a non-empty numpy.ndarray or not a numpy.ndarray.
	Raises:
	This function shouldn’t raise any Exception.
	"""
	if not isinstance(x, np.ndarray) or (x.shape not in [(x.shape[0], 1), (x.shape[0],)]):
		print('Error, x is not valid, it should be a np.ndarray, a vector')
		return None
	x_prime = (x - x.min()) / (x.max() - x.min())
	return x_prime

if __name__ == "__main__":
	print(minmax(np.array([0, 15, -9, 7, 12, 3, -21])))
	print(minmax(np.array([2, 14, -13, 5, 12, 4, -19]).reshape((-1, 1))))
