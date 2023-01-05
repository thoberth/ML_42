import numpy as np

def add_polynomial_features(x, power):
	"""
	Add polynomial features to vector x by raising its values up to the power given in argument. Args:
	x: has to be an numpy.array, a vector of dimension m * 1.
	power: has to be an int, the power up to which the components of vector x are going to be raised.
	Return:
	The matrix of polynomial features as a numpy.array, of dimension m * n,
	containing the polynomial feature values for all training examples.
	None if x is an empty numpy.array.
	None if x or power is not of expected type.
	Raises:
	This function should not raise any Exception.
	"""
	if not (isinstance(x, np.ndarray) and isinstance(power, int)) or \
		x.shape not in [(x.shape[0], 1), (x.shape[0])] or not power > 0:
		print('Error, x must be a vector as a numpy array and power must be a int > 0')
		return None
	X = x
	for i in range(2, power + 1):
		X = np.concatenate((X, x**i), axis = 1)
	return X


if __name__ == "__main__":
	x = np.arange(1, 6).reshape(-1, 1)
	print(add_polynomial_features(x, 3), end='\n\n')
	print(add_polynomial_features(x, 6), end='\n\n')
