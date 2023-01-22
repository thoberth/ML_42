import os, sys
sys.path.append(os.path.abspath("../ex01/"))
from log_pred import logistic_predict_
import numpy as np
import math

def vec_log_loss_(y, y_hat, eps=1e-15):
	"""
	Compute the logistic loss value.
	Args:
	y: has to be an numpy.ndarray, a vector of shape m * 1.
	y_hat: has to be an numpy.ndarray, a vector of shape m * 1.
	eps: epsilon (default=1e-15)
	Returns:
	The logistic loss value as a float.
	None on any error.
	Raises:
	This function should not raise any Exception.
	"""
	log_loss = (1/y.shape[0]) @ (y @ math.log(y_hat) +
                              (np.ones((y.shape[0], 1)) - y) @ math.log(np.ones((y.shape[0], 1)) - y_hat))
	return log_loss

if __name__ == "__main__":
	y1 = np.array([1]).reshape((-1, 1))
	x1 = np.array([4]).reshape((-1, 1))
	theta1 = np.array([[2], [0.5]])
	y_hat1 = logistic_predict_(x1, theta1)
	print(vec_log_loss_(y1, y_hat1))
