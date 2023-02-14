import numpy as np

def reg_log_loss_(y, y_hat, theta, lambda_):
	"""
	Computes the regularized loss of a logistic regression model from two non-empty numpy.ndarray, Args:
		y: has to be an numpy.ndarray, a vector of shape m * 1.
		y_hat: has to be an numpy.ndarray, a vector of shape m * 1.
		theta: has to be a numpy.ndarray, a vector of shape n * 1.
		lambda_: has to be a float.
	Returns:
		The regularized loss as a float.
		None if y, y_hat, or theta is empty numpy.ndarray.
		None if y and y_hat do not share the same shapes.
	Raises:
		This function should not raise any Exception.
	"""
	eps = 1e-15
	theta_prime = np.array(theta)
	theta_prime[0][0] = 0
	v_ones = np.ones((y.shape[0], 1))
	log_loss = - float((y.T.dot(np.log(y_hat + eps)) + (v_ones - y).T.dot(np.log(v_ones - y_hat + eps))) / y.shape[0])
	reg = float(theta_prime.T.dot(theta_prime))
	return log_loss + (lambda_ / (2 * y.shape[0])) * reg

if __name__ == "__main__":
	y = np.array([1, 1, 0, 0, 1, 1, 0]).reshape((-1, 1))
	y_hat = np.array([.9, .79, .12, .04, .89, .93, .01]).reshape((-1, 1))
	theta = np.array([1, 2.5, 1.5, -0.9]).reshape((-1, 1))
	# Example :
	print(reg_log_loss_(y, y_hat, theta, .5))
	# Output: 0.43377043716475955
	# Example :
	print(reg_log_loss_(y, y_hat, theta, .05))
	# Output: 0.13452043716475953
	# Example :
	print(reg_log_loss_(y, y_hat, theta, .9))
	# Output: 0.6997704371647596
