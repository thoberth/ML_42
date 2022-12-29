import numpy as np

def simple_gradient(x, y, theta):
	"""
	Computes a gradient vector from three non-empty numpy.array, without any for loop.
	The three arrays must have compatible shapes.
	Args:
	x: has to be a numpy.array, a matrix of shape m * 1.
	y: has to be a numpy.array, a vector of shape m * 1.
	theta: has to be a numpy.array, a 2 * 1 vector.
	Return:
	The gradient as a numpy.ndarray, a vector of dimension 2 * 1.
	None if x, y, or theta is an empty numpy.ndarray.
	None if x, y and theta do not have compatible dimensions.
	Raises:
	This function should not raise any Exception.
	"""
	if len(x.shape) == 1:
		x = x.reshape(-1, 1)
	# contient la tranposé de : la matrice x concatener un colonne de 1
	X = np.concatenate((np.ones((x.shape[0], 1)), x), axis = 1)
	X_theta = X @ theta # contient les resultats de la prediction, @ -> operateur de produit matricielle
	J_theta = (1 / len(y)) * (X.T.dot((X_theta - y)))
	return J_theta

if __name__ == "__main__":
	x = np.array([12.4956442, 21.5007972, 31.5527382, 48.9145838, 57.5088733]).reshape((-1, 1)) # x : pour entrainer le modele
	y = np.array([37.4013816, 36.1473236, 45.7655287, 46.6793434, 59.5585554]).reshape((-1, 1)) # y : resultats attendus ou target

	print('x :', x, end='\n\n\n')
	print('y :', y, end='\n\n\n')

	theta1 = np.array([2, 0.7]).reshape((-1, 1)) # (a, b)
	print('theta1 :', theta1, end='\n\n\n')
	theta2 = np.array([1, -0.4]).reshape((-1, 1)) # (a, b)
	print(theta2, end='\n\n\n')

	print(simple_gradient(x, y, theta1), end='\n\n')
	print(simple_gradient(x, y, theta2))

	# ADD THE CHECK FOR THE INPUT OF SIMPLE_GRADIENT
