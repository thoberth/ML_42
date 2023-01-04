import math
import numpy as np


def zscore(x):
	"""Computes the normalized version of a non-empty numpy.ndarray using the z-score standardization. Args:
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
	u = np.sum(x) / x.size
	o = math.sqrt(sum([(x_elem - u) ** 2 for x_elem in x]) / x.size - 1)
	x_prime = (x - u) / o
	return x_prime

if __name__ == "__main__":
	print(zscore(np.array([0, 15, -9, 7, 12, 3, -21])))
	print(zscore(np.array([2, 14, -13, 5, 12, 4, -19]).reshape((-1, 1))))
