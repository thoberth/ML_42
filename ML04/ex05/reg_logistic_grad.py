import numpy as np
import math

def predict_(x, theta):
	x = np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)
	return 1 / (1 + math.e ** (-(x @ theta)))

def reg_logistic_grad(y, x, theta, lambda_):
	"""
	Computes the regularized logistic gradient of three non-empty numpy.ndarray, with two for-loops. The three array Args:
		y: has to be a numpy.ndarray, a vector of shape m * 1.
		x: has to be a numpy.ndarray, a matrix of dimesion m * n.
		theta: has to be a numpy.ndarray, a vector of shape n * 1.
		lambda_: has to be a float.
	Returns:
		A numpy.ndarray, a vector of shape n * 1, containing the results of the formula for all j.
		None if y, x, or theta are empty numpy.ndarray.
		None if y, x or theta does not share compatibles shapes.
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

def vec_reg_logistic_grad(y, x, theta, lambda_):
	"""
	Computes the regularized logistic gradient of three non-empty numpy.ndarray, without any for-loop. The three arr Args:
		y: has to be a numpy.ndarray, a vector of shape m * 1.
		x: has to be a numpy.ndarray, a matrix of shape m * n.
		theta: has to be a numpy.ndarray, a vector of shape n * 1.
		lambda_: has to be a float.
	Returns:
		A numpy.ndarray, a vector of shape n * 1, containing the results of the formula for all j.
		None if y, x, or theta are empty numpy.ndarray.
		None if y, x or theta does not share compatibles shapes.
	Raises:
		This function should not raise any Exception.
	"""
	theta_prime = np.array(theta)
	theta_prime[0][0] = 0
	X_prime = np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)
	J = (X_prime.T.dot(predict_(x, theta) - y) +
	     lambda_ * theta_prime) / y.shape[0]
	return J

if __name__ == "__main__":
	x = np.array([[0, 2, 3, 4], [2, 4, 5, 5],
				[1, 3, 2, 7]])
	y=np.array([[0], [1], [1]])
	theta=np.array([[-2.4], [-1.5], [0.3], [-1.4], [0.7]])
	# Example 1.1:
	print(reg_logistic_grad(y, x, theta, 1), end='\n\n')
	# Output:
	# array([[-0.55711039],
	# 		[-1.40334809],
	# 		[-1.91756886],
	# 		[-2.56737958],
	# 		[-3.03924017]])
	# Example 1.2:
	print(vec_reg_logistic_grad(y, x, theta, 1), end='\n\n')
	# Output:
	# array([[-0.55711039],
	# 		[-1.40334809],
	# 		[-1.91756886],
	# 		[-2.56737958],
	# 		[-3.03924017]])
	# Example 2.1:
	print(reg_logistic_grad(y, x, theta, 0.5), end='\n\n')
	# Output:
	# array([[-0.55711039],
	# 		[-1.15334809],
	# 		[-1.96756886],
	# 		[-2.33404624],
	# 		[-3.15590684]])
	# Example 2.2:
	print(vec_reg_logistic_grad(y, x, theta, 0.5), end='\n\n')
	# Output:
	# array([[-0.55711039],
	# 		[-1.15334809],
	# 		[-1.96756886],
	# 		[-2.33404624],
	# 		[-3.15590684]])
	# Example 3.1:
	print(reg_logistic_grad(y, x, theta, 0.0), end='\n\n')
	# Output:
	# array([[-0.55711039],
	# 		[-0.90334809],
	# 		[-2.01756886],
	# 		[-2.10071291],
	# 		[-3.27257351]])
	# Example 3.2:
	print(vec_reg_logistic_grad(y, x, theta, 0.0), end='\n\n')
	# Output:
	# array([[-0.55711039],
	# 		[-0.90334809],
	# 		[-2.01756886],
	# 		[-2.10071291],
	# 		[-3.27257351]])
