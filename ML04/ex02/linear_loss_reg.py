import numpy as np

def reg_loss_(y, y_hat, theta, lambda_):
	"""
	Computes the regularized loss of a linear regression model from two non-empty numpy.array, without any for loop. Args:
		y: has to be an numpy.ndarray, a vector of shape m * 1.
		y_hat: has to be an numpy.ndarray, a vector of shape m * 1.
		theta: has to be a numpy.ndarray, a vector of shape n * 1.
		lambda_: has to be a float.
	Returns:
		The regularized loss as a float.
		None if y, y_hat, or theta are empty numpy.ndarray.
		None if y and y_hat do not share the same shapes.
	Raises:
		This function should not raise any Exception.
	"""
	# A = np.array([[1], [2], [3], [4], [5], [6], [7]])


	# B = np.array([[7], [6], [5], [4], [3], [2], [1]])

	# result = np.dot(A, B.T)
	# print(result)
	theta_prime = np.array(theta)
	theta_prime[0][0] = 0
	loss = float((y_hat - y).T.dot(y_hat - y))
	res = loss + float(lambda_ * (theta_prime.T.dot(theta_prime)))
	return (res/(2*y.shape[0]))

if __name__=="__main__":
	y = np.array([2, 14, -13, 5, 12, 4, -19]).reshape((-1, 1))
	y_hat = np.array([3, 13, -11.5, 5, 11, 5, -20]).reshape((-1, 1))
	theta = np.array([1, 2.5, 1.5, -0.9]).reshape((-1, 1))
	# Example :
	print(reg_loss_(y, y_hat, theta, .5))
	# Output: 0.8503571428571429
	# Example :
	print(reg_loss_(y, y_hat, theta, .05))
	# Output: 0.5511071428571429
	# Example :
	print(reg_loss_(y, y_hat, theta, .9))
	# Output: 1.116357142857143
