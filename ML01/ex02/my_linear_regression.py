import numpy as np

class MyLinearRegression:
	"""
	Description:
	My personnal linear regression class to fit like a boss.
	"""
	def __init__(self, thetas, alpha=0.001, max_iter=1000):
		self.alpha = alpha
		self.max_iter = max_iter
		self.thetas = thetas

	def fit_(self, x, y):
		X = np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)
		for _ in range(self.max_iter):
			# X_theta contient les resultats de la prediction, @ -> operateur de produit matricielle
			X_theta = self.predict_(X)
			J_theta = (1 / len(y)) * (X.T.dot((X_theta - y)))
			self.thetas = self.thetas - (self.alpha * J_theta)
		return (self.thetas)

	def predict_(self, x):
		if x.shape != (x.shape[0], 2):
			x = np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)
		x = x @ self.thetas
		return x

	def loss_elem_(self, y, y_hat):
		squared_error = (y_hat - y) ** 2
		return squared_error

	def loss_(self, y, y_hat):
		squared_error = self.loss_elem_(y, y_hat)
		sum_squared_error = sum(squared_error)
		return sum_squared_error / (y.size*2)


if __name__ == "__main__":
	x = np.array([[12.4956442], [21.5007972], [31.5527382], [48.9145838], [57.5088733]])
	y = np.array([[37.4013816], [36.1473236], [45.7655287], [46.6793434], [59.5585554]])
	lr1 = MyLinearRegression(np.array([[2], [0.7]]))

	y_hat = lr1.predict_(x)
	print(y_hat,end='\n\n')
	print(lr1.loss_elem_(y, y_hat), end='\n\n')
	print(lr1.loss_(y, y_hat), end='\n\n')

	lr2 = MyLinearRegression(np.array([[1], [1]]), 5e-8, 1500000)
	print(lr2.fit_(x, y), end='\n\n')
	y_hat = lr2.predict_(x)
	print(y_hat, end='\n\n')
	print(lr2.loss_elem_(y, y_hat), end='\n\n')
	print(lr2.loss_(y, y_hat), end='\n\n')
