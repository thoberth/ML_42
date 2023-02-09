import os
import sys
import math
import numpy as np


class MyLogisticRegression():
	"""
	Description:
			My personnal logistic regression to classify things.
	"""

	def __init__(self, theta, alpha=0.001, max_iter=1000):
		self.alpha = alpha
		self.max_iter = max_iter
		self.theta = theta

	def predict_(self, x):
		if not (isinstance(x, np.ndarray) and isinstance(self.theta, np.ndarray)) or\
		(x.shape not in [(x.shape[0],), (x.shape[0], self.theta.shape[0] - 1)]) or\
                        (self.theta.shape not in [(self.theta.shape[0],), (self.theta.shape[0], 1)]):
			print('logistic predict function error in parameters')
			return None
		x = np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)
		y_hat = np.ones((x.shape[0], 1))
		for i in range(x.shape[0]):
			y_hat[i][0] = (1. / (1. + math.exp(-(x[i] @ self.theta))))
		return y_hat

	def vec_log_gradient(self, x, y):
		X = np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)
		log_gradient = (1/len(y)) * X.T.dot(self.predict_(x) - y)
		return log_gradient

	def fit_(self, x, y):
		for _ in range(self.max_iter):
			log_gradient = self.vec_log_gradient(x, y)
			self.theta = self.theta - (self.alpha * log_gradient)
		return self.theta

	def loss_elem_(self, y, y_hat, eps=1e-15):
		log_loss = (y * np.log(y_hat + eps)) + ((1 - y)
												* np.log(1 - y_hat + eps))
		return log_loss

	def loss_(self, y, y_hat, eps=1e-15):
		log_loss = (y * np.log(y_hat + eps)) +\
			((1 - y) * np.log(1 - y_hat + eps))
		return (-1/len(y)) * np.sum(log_loss)


if __name__ == "__main__":
	MyLR = MyLogisticRegression
	X = np.array([[1., 1., 2., 3.], [5., 8., 13., 21.], [3., 5., 9., 14.]])
	Y = np.array([[1], [0], [1]])
	thetas = np.array([[2], [0.5], [7.1], [-4.3], [2.09]])
	mylr = MyLR(thetas)

	# Example 0:
	y_hat = mylr.predict_(X)
	print(y_hat)
	# Output:
	# array([[0.99930437],
	# 	[1.],
	# 	[1.]])

	# Example 1:
	print(mylr.loss_(Y, y_hat))
	# Output:
	# 11.513157421577004

	# Example 2:
	print(mylr.fit_(X, Y))
	# Output:
	# array([[2.11826435]
	# 	[0.10154334]
	# 	[6.43942899]
	# 	[-5.10817488]
	# 	[0.6212541]])

	# Example 3:
	y_hat2 = mylr.predict_(X)
	print(y_hat2)
	# Output:
	# array([[0.57606717]
	# 	[0.68599807]
	# 	[0.06562156]])

	# Example 4:
	print(mylr.loss_(Y, y_hat2))
	# Output:
	# 1.4779126923052268
