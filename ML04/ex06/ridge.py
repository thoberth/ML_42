import numpy as np
import pandas as pd

class MyRidge():
	"""
	Description:
		My personnal ridge regression class to fit like a boss.
	"""
	def __init__(self, thetas, alpha=0.001, max_iter=1000,
			lambda_=0.5):
		self.alpha = alpha
		self.max_iter = max_iter
		self.thetas = thetas
		self.lambda_ = lambda_

	def set_params_(self, **kwargs):
		for key, value in kwargs.items():
			if key == 'thetas':
				self.thetas = value
			elif key == 'alpha':
				self.alpha = value
			elif key == 'max_iter':
				self.max_iter = value
			elif key == 'lambda_':
				self.lambda_ = value
			else : print('Error Dict Key is unknow')

	def get_params_(self):
		return self.thetas, self.alpha, self.max_iter, self.lambda_

	def predict_(self, x):
		X = np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)
		return X @ self.thetas

	def gradient_(self, x, y):
		theta_prime = np.array(self.thetas)
		theta_prime[0][0] = 0
		X_prime = np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)
		J = (X_prime.T.dot(self.predict_(x) - y) + self.lambda_ * theta_prime) / y.shape[0]
		return J

	def fit_(self, x, y):
		for _ in range(self.max_iter):
			gradient = self.gradient_(x, y)
			self.thetas = self.thetas - (self.alpha * gradient)
		return self.thetas

	def loss_elem_(self, y, y_hat):
		theta_prime = np.array(self.thetas)
		theta_prime[0][0] = 0
		loss = float((y_hat - y).T.dot(y_hat - y))
		res = loss + float(self.lambda_ * (theta_prime.T.dot(theta_prime)))
		return (res)

	def loss_(self, y, y_hat):
		return self.loss_elem_(y, y_hat) / (2 * y.shape[0])

if __name__ == "__main__":
	X = np.array([[1., 1., 2., 3.], [5., 8., 13., 21.], [34., 55., 89., 144.]])
	Y = np.array([[23.], [48.], [218.]])
	mylr = MyRidge(np.array([[1.], [1.], [1.], [1.], [1]]))
	y_hat = mylr.predict_(X)
	print('Les valeurs reelles sont 23, 48, 218\nPrediction avant entrainement:')
	print('Y_hat =', y_hat)
	# print(mylr.loss_elem_(Y, y_hat))
	print('La perte est de :', mylr.loss_(Y, y_hat))
	mylr.alpha = 1.6e-4
	mylr.max_iter = 200000
	print('On entraine notre modele..........')
	mylr.fit_(X, Y)
	# print(mylr.thetas)
	y_hat = mylr.predict_(X)
	print('Y_hat est maintenant egale a :', y_hat)
	# print(mylr.loss_elem_(Y, y_hat))
	print('La perte est de:', mylr.loss_(Y, y_hat))