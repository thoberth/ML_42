import numpy as np

def fit_(x, y, theta, alpha, max_iter):
	"""
	Description:
	Fits the model to the training dataset contained in x and y.
	Args:
	x: has to be a numpy.ndarray, a vector of dimension m * 1: (number of training examples, 1).
	y: has to be a numpy.ndarray, a vector of dimension m * 1: (number of training examples, 1).
	theta: has to be a numpy.ndarray, a vector of dimension 2 * 1.
	alpha: has to be a float, the learning rate
	max_iter: has to be an int, the number of iterations done during the gradient descent
	Returns:
	new_theta: numpy.ndarray, a vector of dimension 2 * 1.
	None if there is a matching dimension problem.
	Raises:
	This function should not raise any Exception.
	"""
	# contient la matrice x concatener un colonne de 1
	X = np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)
	for _ in range(max_iter):
		# contient les resultats de la prediction, @ -> operateur de produit matricielle
		X_theta = X @ theta
		J_theta = (1 / len(y)) * (X.T.dot((X_theta - y)))
		theta = theta - (alpha * J_theta)
	return(theta)


def predict_(x, theta):
	y_hat = theta[0] + theta[1] * x
	return y_hat


if __name__ == "__main__":
	x = np.array([[12.4956442], [21.5007972], [31.5527382], [48.9145838], [57.5088733]])
	y = np.array([[37.4013816], [36.1473236], [45.7655287], [46.6793434], [59.5585554]])
	theta= np.array([1, 1]).reshape((-1, 1))

	print('x :', x, end='\n\n\n')
	print('y :', y, end='\n\n\n')
	print('theta :', theta, end='\n\n\n')
	theta2 = fit_(x, y, theta, 5e-8, 1500000)
	print(theta2, end='\n\n\n')
	print(predict_(x, theta2))


# ADD THE CHECK FOR THE INPUT OF FIT
