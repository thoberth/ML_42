import os, sys
from parent_class import MyLogisticRegression as ParentMLR
import numpy as np

class MyLogisticRegression(ParentMLR):
	"""
		Description:
		My personnal logistic regression to classify things.
	"""
	# We consider l2 penality only. One may wants to implement other penalities
	supported_penalities = ['l2']

	# Check on type, data type, value ... if necessary
	def __init__(self, theta, alpha=0.001, max_iter=1000, penalty='l2', lambda_=1.0):
		super().__init__(theta, alpha, max_iter)
		self.alpha = alpha
		self.max_iter = max_iter
		self.theta = theta
		self.penalty = penalty
		self.lambda_ = lambda_ if penalty in self.supported_penalities else 0

	def gradient_(self, x, y):
		theta_prime = np.array(self.theta)
		theta_prime[0][0] = 0
		X_prime = np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)
		J = (X_prime.T.dot(super().predict_(x) - y) +
			lambda_ * theta_prime) / y.shape[0]
		return J

	def fit_(self, x, y):
		if self.penalty != 'l2':
			return super().fit_(x, y)
		else:
			for _ in range(self.max_iter):
				log_gradient = self.gradient_(x, y)
				self.theta = self.theta - (self.alpha * log_gradient)
			return self.theta


if __name__ == "__main__":
	mylogr = MyLogisticRegression
	theta = np.array([[-2.4], [-1.5], [0.3], [-1.4], [0.7]])
	# Example 1:
	model1 = mylogr(theta, lambda_=5.0)
	print(model1.penalty)
	# Output 'l2'
	print(model1.lambda_)
	# Output 5.0


	# Example 2:
	model2 = mylogr(theta, penalty=None)
	print(model2.penalty)
	# Output None
	print(model2.lambda_)
	# Output 0.0


	# Example 3:
	model3 = mylogr(theta, penalty=None, lambda_=2.0)
	print(model3.penalty)
	# Output None
	print(model3.lambda_)
	# Output 0.0
