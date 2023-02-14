import numpy as np

def predict_(x, theta):
	X = np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)
	return X @ theta

def reg_linear_grad(y, x, theta, lambda_):
	"""
	Computes the regularized linear gradient of three non-empty numpy.ndarray,
		with two for-loop. The three arrays must have compatible shapes.
	Args:
		y: has to be a numpy.ndarray, a vector of shape m * 1.
		x: has to be a numpy.ndarray, a matrix of dimesion m * n.
		theta: has to be a numpy.ndarray, a vector of shape (n + 1) * 1.
		lambda_: has to be a float.
	Return:
		A numpy.ndarray, a vector of shape (n + 1) * 1, containing the results of the formula for all j.
		None if y, x, or theta are empty numpy.ndarray.
		None if y, x or theta does not share compatibles shapes.
		None if y, x or theta or lambda_ is not of the expected type.
	Raises:
		This function should not raise any Exception.
	"""
	J = np.zeros(theta.shape)
	for xi, yi, y_hati in zip(x, y, predict_(x, theta)):
		J[0] += (y_hati - yi)
	for xi, yi, y_hati in zip(x, y, predict_(x, theta)):
		J[1:] += (y_hati - yi) * xi.reshape(-1, 1) 
	J[1:] += lambda_ * theta[1:]
	return J / y.shape[0]

def vec_reg_linear_grad(y, x, theta, lambda_):
	"""
	Computes the regularized linear gradient of three non-empty numpy.ndarray,
		without any for-loop. The three arrays must have compatible shapes.
	Args:
		y: has to be a numpy.ndarray, a vector of shape m * 1.
		x: has to be a numpy.ndarray, a matrix of dimesion m * n.
		theta: has to be a numpy.ndarray, a vector of shape (n + 1) * 1.
		lambda_: has to be a float.
	Return:
		A numpy.ndarray, a vector of shape (n + 1) * 1, containing the results of the formula for all j.
		None if y, x, or theta are empty numpy.ndarray.
		None if y, x or theta does not share compatibles shapes.
		None if y, x or theta or lambda_ is not of the expected type.
	Raises:
		This function should not raise any Exception.
	"""
	theta_prime = np.array(theta)
	theta_prime[0][0] = 0
	X_prime = np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)
	J = (X_prime.T.dot(predict_(x, theta) - y) + lambda_ * theta_prime) / y.shape[0]
	return J

if __name__ == "__main__":
	x = np.array([
				[-6, -7, -9], [13, -2, 14], [-7, 14, -1], [-8, -4, 6], [-5, -9, 6], [1, -5, 11], [9, -11, 8]])
	y = np.array([[2], [14], [-13], [5], [12], [4], [-19]])
	theta = np.array([[7.01], [3], [10.5], [-6]])
	# Example 1.1:
	print(reg_linear_grad(y, x, theta, 1), end='\n\n')
	# Output:
	# array([[-60.99],
	# 	[-195.64714286],
	# 	[863.46571429],
	# 	[-644.52142857]])
	# Example 1.2:
	print(vec_reg_linear_grad(y, x, theta, 1), end='\n\n')
	# Output:
	# array([[-60.99],
	# 	[-195.64714286],
	# 	[863.46571429],
	# 	[-644.52142857]])
	# Example 2.1:
	print(reg_linear_grad(y, x, theta, 0.5), end='\n\n')
	# Output:
	# array([[-60.99],
	# 	[-195.86142857],
	# 	[862.71571429],
	# 	[-644.09285714]])
	# Example 2.2:
	print(vec_reg_linear_grad(y, x, theta, 0.5), end='\n\n')
	# Output:
	# array([[-60.99],
	# 	[-195.86142857],
	# 	[862.71571429],
	# 	[-644.09285714]])
	# Example 3.1:
	print(reg_linear_grad(y, x, theta, 0.0), end='\n\n')
	# Output:
	# array([[-60.99],
	# 	[-196.07571429],
	# 	[861.96571429],
	# 	[-643.66428571]])
	# Example 3.2:
	print(vec_reg_linear_grad(y, x, theta, 0.0))
	# Output:
	# array([[-60.99],
	# 	[-196.07571429],
	# 	[861.96571429],
	# 	[-643.66428571]])
