import numpy as np

def add_polynomial_features(x, power):
	"""Add polynomial features to matrix x by raising its columns to every power in the range of 1 up to the power give Args:
					x: has to be an numpy.ndarray, a matrix of shape m * n.
					power: has to be an int, the power up to which the columns of matrix x are going to be raised.
			Returns:
					The matrix of polynomial features as a numpy.ndarray, of shape m * (np), containg the polynomial feature va
					None if x is an empty numpy.ndarray.
			Raises:
	This function should not raise any Exception.
	"""
	X = x
	for y in range(2, power + 1):
		# print(X[:, -x.shape[1]:])
		X = np.concatenate((X, X[:, :x.shape[1]]**y), axis = 1)
	return X

if __name__ == "__main__":
	x = np.arange(1, 11).reshape(5, 2)
	# Example 1:
	print(add_polynomial_features(x, 3))

	# Example 2:
	print(add_polynomial_features(x, 4))
