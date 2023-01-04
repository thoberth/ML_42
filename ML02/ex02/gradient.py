import numpy as np

def gradient(x, y, theta):
	"""
	Computes a gradient vector from three non-empty numpy.array, without any for-loop. The three arrays must have the compatible dimensions.
	Args:
	x: has to be an numpy.array, a matrix of dimension m * n.
	y: has to be an numpy.array, a vector of dimension m * 1.
	theta: has to be an numpy.array, a vector (n +1) * 1.
	Return:
	The gradient as a numpy.array, a vector of dimensions n * 1,
	containg the result of the formula for all j.
	None if x, y, or theta are empty numpy.array.
	None if x, y and theta do not have compatible dimensions.
	None if x, y or theta is not of expected type.
	Raises:
	This function should not raise any Exception.
	"""
	X = np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)
	# contient les resultats de la prediction, @ -> operateur de produit matricielle
	X_theta = X @ theta
	J_theta = (1 / len(y)) * (X.T.dot((X_theta - y)))
	return J_theta

if __name__ == "__main__":
	x = np.array([
			  [-6,  -7,  -9],
			  [13,  -2,  14],
			  [-7, 14, -1], [-8, -4, 6], [-5, -9, 6], [1, -5, 11], [9, -11, 8]])
	y = np.array([2, 14, -13, 5, 12, 4, -19]).reshape((-1, 1))
	theta1 = np.array([0.0, 3, 0.5, -6]).reshape((-1, 1))  # <-- missing `0`

	print(gradient(x, y, theta1))

	theta2 = np.array([0.0, 0, 0, 0]).reshape((-1, 1))  # <-- missing `0`
	print(gradient(x, y, theta2))
