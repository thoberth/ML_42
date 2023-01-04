import numpy as np

class MyLinearRegression:
	"""
	Description:
	My personnal linear regression class to fit like a boss.
	"""

	def __init__(self, theta, alpha=0.001, max_iter=1000):
		self.alpha = alpha
		self.max_iter = max_iter
		self.theta = theta

	def fit_(self, x, y):
		X = np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)
		for _ in range(self.max_iter):
			# X_theta contient les resultats de la prediction, @ -> operateur de produit matricielle
			X_theta = self.predict_(X)
			J_theta = (1 / len(y)) * (X.T.dot((X_theta - y)))
			self.theta = self.theta - (self.alpha * J_theta)
		return (self.theta)

	def predict_(self, x):
		if x.shape != (x.shape[0], len(self.theta)):
			x = np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)
		x = x @ self.theta
		return x

	def loss_elem_(self, y, y_hat):
		squared_error = (y_hat - y) ** 2
		return squared_error

	def loss_(self, y, y_hat):
		squared_error = self.loss_elem_(y, y_hat)
		sum_squared_error = sum(squared_error)
		return sum_squared_error / (y.size*2)

if __name__=="__main__":
	X = np.array([[1., 1., 2., 3.], [5., 8., 13., 21.], [34., 55., 89., 144.]])
	Y = np.array([[23.], [48.], [218.]])
	mylr = MyLinearRegression([[1.], [1.], [1.], [1.], [1]])

	y_hat = mylr.predict_(X)
	print(y_hat, end='\n\n')

	print(mylr.loss_elem_(Y, y_hat), end='\n\n')

	print(mylr.loss_(Y, y_hat), end='\n\n')

	mylr.alpha = 1.6e-4
	mylr.max_iter = 200000
	mylr.fit_(X, Y)
	print(mylr.theta, end='\n\n')

	y_hat = mylr.predict_(X)
	print(y_hat, end='\n\n')

	print(mylr.loss_elem_(Y, y_hat), end='\n\n')

	print(mylr.loss_(Y, y_hat), end='\n\n')
