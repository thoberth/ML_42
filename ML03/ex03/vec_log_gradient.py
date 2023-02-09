import os, sys
sys.path.append(os.path.abspath("../ex01/"))
from log_pred import *
import numpy as np

def vec_log_gradient(x, y, theta):
	"""
	Computes a gradient vector from three non-empty numpy.ndarray, without any for-loop.
	The three arrays must have comp
	Args:
		x: has to be an numpy.ndarray, a matrix of shape m * n.
		y: has to be an numpy.ndarray, a vector of shape m * 1.
		theta: has to be an numpy.ndarray, a vector (n +1) * 1.
	Returns:
		The gradient as a numpy.ndarray, a vector of shape n * 1,
		contain the result of the formula for all j.
		None if x, y, or theta are empty numpy.ndarray.
		None if x, y and theta do not have compatible shapes.
	Raises:
		This function should not raise any Exception.
	"""
	X = np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)
	log_gradient = (1/len(y)) * X.T.dot(logistic_predict_(x, theta) - y)
	return log_gradient

if __name__ == "__main__":
	y1 = np.array([1]).reshape((-1, 1))
	x1 = np.array([4]).reshape((-1, 1))
	theta1 = np.array([[2], [0.5]])
	print(vec_log_gradient(x1, y1, theta1), end = '\n\n')
	# Output:
	# array([[-0.01798621],
	# 	[-0.07194484]])

	# Example 2:
	y2 = np.array([[1], [0], [1], [0], [1]])
	x2 = np.array([[4], [7.16], [3.2], [9.37], [0.56]])
	theta2 = np.array([[2], [0.5]])
	print(vec_log_gradient(x2, y2, theta2), end='\n\n')
	# Output:
	# array([[0.3715235],
	# 	[3.25647547]])

	# Example 3:
	y3 = np.array([[0], [1], [1]])
	x3 = np.array([[0, 2, 3, 4], [2, 4, 5, 5], [1, 3, 2, 7]])
	theta3 = np.array([[-2.4], [-1.5], [0.3], [-1.4], [0.7]])
	print(vec_log_gradient(x3, y3, theta3), end='\n\n')
	# Output:
	# array([[-0.55711039],
	# 	[-0.90334809],
	# 	[-2.01756886],
	# 	[-2.10071291],
	# 	[-3.27257351]])
